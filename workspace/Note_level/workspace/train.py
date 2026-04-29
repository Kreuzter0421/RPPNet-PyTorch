import argparse
import yaml
import torch
from model.model import MidiDataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from model.model import NoteTransformer
import os
from tqdm import tqdm
import glob
import datetime
from torch.nn.utils import clip_grad_norm_

def collate_noteLevel(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        values = [item[key] for item in batch]
        if key in ('name', 'n_in_sequences'):
            collated[key] = values
        else:
            collated[key] = default_collate(values)
    return collated


def build_feature_loss_masks(feature_names, first_note_mask):
    if first_note_mask is None:
        return None
    mask_tokens = first_note_mask
    if mask_tokens.dtype != torch.bool:
        mask_tokens = mask_tokens > 0
    # Align mask with gt (which is shifted by one position relative to input tokens)
    mask_gt = torch.zeros_like(mask_tokens, dtype=torch.bool)
    if mask_gt.shape[1] > 1:
        mask_gt[:, :-1] = mask_tokens[:, 1:]
    feature_masks = []
    all_true = torch.ones_like(mask_gt, dtype=torch.bool)
    for name in feature_names:
        if name in ('bar', 'position'):
            feature_masks.append(~mask_gt)
        else:
            feature_masks.append(all_true)
    return feature_masks


def parse_args():
    parser = argparse.ArgumentParser(description="NoteTransformer training")
    parser.add_argument("--config", type=str,
                        default='../config/config.yaml',
                        help="Path to the YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint containing model and optimizer states to resume from")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model weights (state_dict) to initialize from")
    parser.add_argument("--model-save-dir", type=str, default=None,
                        help="Optional override for model saving directory; defaults to config path or resume directory")
    return parser.parse_args()



def evaluate(e, cfg, model, val_iter, device):
    model.eval()    
    with torch.no_grad():
        loss_list = []
        for b, batch in enumerate(val_iter):

            V = batch['rpp_feat'].to(device)
            tgt = batch['note_feat'].to(device)
            tgt_gt = batch['note_feat_gt'].to(device)
            tgt_mask = batch['note_mask'].to(device)
            if False:
                adj = None
            first_note_mask = batch.get('first_note_mask')
            feature_masks = None
            if first_note_mask is not None:
                feature_masks = build_feature_loss_masks(cfg['note_feature_selected'], first_note_mask.to(device))

            # Forward
            out = model(V=V,tgt=tgt, tgt_mask=tgt_mask)

            # Loss
            loss = model.loss(predict=out, gt=tgt_gt, feature_masks=feature_masks)

            # clip_grad_norm_(model.parameters(), grad_clip)

            # Save loss
            loss_list.append(loss.data.item())

    return sum(loss_list) / len(loss_list)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train(e, cfg, model, optimizer, train_iter, device, scheduler=None):
    model.train()
    loss_list = []    
    grad_clip = float(cfg.get('grad_clip', 0.0) or 0.0)
    for b, batch in enumerate(train_iter):

        V = batch['rpp_feat'].to(device)
        tgt = batch['note_feat'].to(device)
        tgt_gt = batch['note_feat_gt'].to(device)
        tgt_mask = batch['note_mask'].to(device)
        if False:
            adj = None
        first_note_mask = batch.get('first_note_mask')
        feature_masks = None
        if first_note_mask is not None:
            feature_masks = build_feature_loss_masks(cfg['note_feature_selected'], first_note_mask.to(device))

        decoder_inputs = tgt
        out = model(V=V, tgt=decoder_inputs, tgt_mask=tgt_mask)

        # Loss
        optimizer.zero_grad()
        loss = model.loss(predict=out, gt=tgt_gt, feature_masks=feature_masks)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss detected: {loss.item()}")
            print(f"RPP Feat Stats: Min={V.min()}, Max={V.max()}, HasZero={(V==0).any()}, Shape={V.shape}")
            print(f"Note Feat Stats: Min={tgt.min()}, Max={tgt.max()}, HasZero={(tgt==0).any()}, Shape={tgt.shape}")
            raise ValueError("Training interrupted due to NaN/Inf loss")

        # Backward
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        # clip_grad_norm_(model.parameters(), grad_clip)

        # Save loss
        loss_list.append(loss.data.item())

        # print loss
        if b % 100 == 0 and b != 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            print(f"iter({b // 100}) train_loss:{loss.data.item()} | lr:{current_lr:.2e}")

    return sum(loss_list) / len(loss_list)


def main(args=None):
    args = parse_args() if args is None else args
    if args.resume and args.pretrained:
        raise ValueError("--resume and --pretrained cannot be used together")

    # Config
    with open(args.config, 'r') as f:
        cfg = yaml.full_load(f)

    if cfg['usegpu']:
        device = torch.device("cuda:{}".format(cfg['gpuID']) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # Log_Dir
    date = datetime.datetime.now()
    date_str = f'{date.month}-{date.day}-{date.hour}-{date.minute}'
    default_save_dir = os.path.join(cfg['model_save_dir'], date_str)
    if args.model_save_dir:
        model_save_dir = args.model_save_dir
    elif args.resume:
        model_save_dir = os.path.dirname(args.resume)
    else:
        model_save_dir = default_save_dir
    os.makedirs(model_save_dir, exist_ok=True)
    log_path = os.path.join(model_save_dir, 'log.txt')

    if not args.resume:
        with open(log_path, 'a') as f:
            f.write(f"* Exp Detail:{cfg['exp_detail']}\n")
            f.write(f"* N_head:{cfg['n_head']}\n")
            f.write(f"* Num_layers:{cfg['num_layers']}\n")
    else:
        with open(log_path, 'a') as f:
            f.write(f"# Resume {datetime.datetime.now().isoformat()} from {args.resume}\n")

    # Model
    model = NoteTransformer(cfg=cfg).to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("* Number of parameters: %.2fM *" % (total / 1e6))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # Data
    train_dataset = MidiDataset(config=cfg,dataroot=cfg['train_pkl'])
    val_dataset = MidiDataset(config=cfg,dataroot=cfg['val_pkl'])

    num_workers = int(cfg.get('num_workers', 0))
    pin_memory = bool(cfg.get('pin_memory', device.type == 'cuda'))
    persistent_workers = bool(cfg.get('persistent_workers', False)) and num_workers > 0
    prefetch_factor = cfg.get('prefetch_factor', None) if num_workers > 0 else None

    # Training dataloader: shuffle and (optionally) drop last partial batch for stable batch sizes
    train_dataloader_kwargs = dict(batch_size=cfg['batch_size'],
                                   drop_last=True,
                                   collate_fn=collate_noteLevel,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   persistent_workers=persistent_workers)
    # Validation dataloader: do NOT shuffle and do NOT drop_last so validation covers all samples
    val_dataloader_kwargs = dict(batch_size=cfg['batch_size'],
                                 drop_last=False,
                                 collate_fn=collate_noteLevel,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 persistent_workers=persistent_workers)

    if prefetch_factor is not None:
        train_dataloader_kwargs['prefetch_factor'] = prefetch_factor
        val_dataloader_kwargs['prefetch_factor'] = prefetch_factor

    train_dataloader = DataLoader(dataset=train_dataset, **train_dataloader_kwargs)
    val_dataloader = DataLoader(dataset=val_dataset, **val_dataloader_kwargs)

    # Small sanity log to help debug abrupt val loss changes
    # print number of batches for quick check
    try:
        from math import ceil
        print(f"Train batches: {ceil(len(train_dataset)/cfg['batch_size'])}, Val batches: {ceil(len(val_dataset)/cfg['batch_size'])}")
    except Exception:
        pass
        
    num_training_steps = len(train_dataloader) * cfg['num_epochs']
    num_warmup_steps = 2000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    start_epoch = 0
    best_loss = None

    def load_state_dict_into_model(state):
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
            return state
        model.load_state_dict(state)
        return {
            'epoch': -1,
            'best_loss': None
        }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        metadata = load_state_dict_into_model(checkpoint)
        if 'optimizer_state_dict' in metadata:
            optimizer.load_state_dict(metadata['optimizer_state_dict'])
        best_loss = metadata.get('best_loss', None)
        start_epoch = metadata.get('epoch', -1) + 1
        print(f"Resumed training from {args.resume} (start epoch {start_epoch})")
    elif args.pretrained:
        pretrained_state = torch.load(args.pretrained, map_location=device)
        load_state_dict_into_model(pretrained_state)
        print(f"Loaded pretrained weights from {args.pretrained}")

    if start_epoch >= cfg['num_epochs']:
        print("Training already completed for the requested number of epochs.")
        return

    for epoch in tqdm(range(start_epoch, cfg['num_epochs']), initial=start_epoch, total=cfg['num_epochs']):
        epoch = epoch + 0
        # train
        train_loss = train(epoch, cfg, model, optimizer, train_dataloader, device, scheduler=scheduler)

        # evaluate
        val_loss = evaluate(e=epoch, cfg=cfg, model=model, val_iter=val_dataloader, device=device)

        current_lr = optimizer.param_groups[0]['lr']
        # print
        print(f'Epoch {epoch} train_loss:{train_loss} val_loss:{val_loss} lr:{current_lr}')

        # Save the model if the validation loss is the best we've seen so far.
        if not best_loss or val_loss < best_loss:
            best_loss = val_loss
            print("[!] saving model...")

            # Model Clean
            files = glob.glob(os.path.join(model_save_dir, '*_val_best.pt'))
            for file in files:
                os.remove(file)

            # Save Model Val
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'NoteTransformer_val_best.pt'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': cfg
            }, os.path.join(model_save_dir, 'checkpoint_val_best.pt'))

        # Save Model Val / checkpoint
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'NoteTransformer_train_best.pt'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'config': cfg
        }, os.path.join(model_save_dir, 'checkpoint_last.pt'))
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'epoch_{epoch}.pt'))

        # log
        with open(log_path, 'a') as f:
            f.write(f'Epoch:{epoch:<5} train_loss:{train_loss:<20} val_loss:{val_loss:<20} best_loss:{best_loss:<20}\n')


if __name__ == '__main__':
    main()
# for data in train_dataloader:
#     print('---------------------------------------------------------')
#     for k, v in data.items():
#         if isinstance(v, list):
#             print(v)
#         else:
#             print(k, '  shape:', v.shape)
#
#     # # -------------------- [Test Encoder] --------------------#
#     # encoder = EncoderGAT(cfg=cfg)
#     # V = data['rpp_feature_seq']
#     # E = data['rpp_edge_seq']
#     # out = encoder(V,E)
#     # print(out.shape)
#
#     # # -------------------- [Test Dncoder] --------------------#
#     # decoder = DecoderTransformer(cfg=cfg)
#     # memory = torch.rand((2,128,512))
#     # out = decoder(tgt=data['note_seq'],memory = memory,tgt_mask = data['note_mask'])
#     # print(out,out.shape)
#
#     # -------------------- [Test NoteTransformer] --------------------#
#     model= NoteTransformer(cfg=cfg)
#     out = model(V=data['rpp_feature_seq'],E=data['rpp_edge_seq'],tgt=data['note_seq'],tgt_mask=data['note_mask'])
#     print(out,out.shape)
#     loss = model.loss(predict=out,gt=data['note_seq_gt'])
#     print(loss)