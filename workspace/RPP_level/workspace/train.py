import argparse
import yaml
import torch
from model import MuGraphDataset as MuGraphDataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from model import GraphTransformer
import os
import pickle
from tqdm import tqdm
import glob
import datetime
from torch.nn.utils import clip_grad_norm_
import random

def create_scheduler(optimizer, cfg, num_epochs, steps_per_epoch):
    scheduler_cfg = cfg.get('lr_scheduler')
    if not scheduler_cfg:
        return None, None

    scheduler_type = str(scheduler_cfg.get('type', '')).lower()

    if scheduler_type == 'one_cycle':
        if steps_per_epoch is None or steps_per_epoch <= 0:
            raise ValueError('OneCycleLR requires a positive number of steps per epoch')
        base_lr = float(cfg.get('lr', 1e-4))
        max_lr = float(scheduler_cfg.get('max_lr', base_lr))
        pct_start = float(scheduler_cfg.get('pct_start', 0.1))
        div_factor = float(scheduler_cfg.get('div_factor', 25.0))
        final_div_factor = float(scheduler_cfg.get('final_div_factor', 1e3))
        anneal_strategy = scheduler_cfg.get('anneal_strategy', 'cos')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=safe_final_div_factor(final_div_factor),
            anneal_strategy=anneal_strategy,
            cycle_momentum=scheduler_cfg.get('cycle_momentum', False)
        )
        return scheduler, 'batch'

    if scheduler_type == 'cosine':
        t_0 = int(scheduler_cfg.get('t0', max(1, num_epochs // 4)))
        t_mult = int(scheduler_cfg.get('t_mult', 1))
        base_lr = float(cfg.get('lr', 1e-4))
        eta_min = float(scheduler_cfg.get('eta_min', base_lr * 0.1))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=t_mult,
            eta_min=eta_min
        )
        return scheduler, 'epoch'

    if scheduler_type == 'plateau':
        factor = float(scheduler_cfg.get('factor', 0.5))
        patience = int(scheduler_cfg.get('patience', 5))
        threshold = float(scheduler_cfg.get('threshold', 1e-4))
        min_lr = float(scheduler_cfg.get('min_lr', 0.0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get('mode', 'min'),
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )
        return scheduler, 'val'

    print(f"[Scheduler] Unknown scheduler type '{scheduler_type}', skipping scheduler setup.")
    return None, None


def safe_final_div_factor(value):
    # guard extremely small values while keeping backwards compatibility
    return max(float(value), 1.0)

def collate_feature(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        values = [item[key] for item in batch]
        if key in ('name', 'n_in_sequences'):
            collated[key] = values
        else:
            collated[key] = default_collate(values)
    return collated


def parse_args():
    parser = argparse.ArgumentParser(description="Seq2Graph Feature training")
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


def sync_rps_feature_dict(cfg, config_path):
    dict_rel_path = cfg.get('rps_feat2idx_path')
    if not dict_rel_path:
        return cfg
    config_dir = os.path.dirname(os.path.abspath(config_path))
    dict_path = dict_rel_path if os.path.isabs(dict_rel_path) else os.path.normpath(os.path.join(config_dir, dict_rel_path))
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"rps_feat2idx_path not found: {dict_path}")
    with open(dict_path, 'rb') as f:
        vocab = pickle.load(f)
    dims = {k: len(v) for k, v in vocab.items()}
    feature_dict = cfg.setdefault('rps_feature_dict', {})
    selected = cfg.get('rps_feature_selected', []) or []
    missing = [feat for feat in selected if feat not in dims and feat != 'global_pos']
    if missing:
        raise KeyError(f"Features {missing} missing from {dict_path}")
    for feat in selected:
        if feat in dims:
            feature_dict[feat] = dims[feat]
    if 'cadence_tag' in cfg.get('rps_feature_all', []) and 'cadence_tag' in dims:
        feature_dict['cadence_tag'] = dims['cadence_tag']
    return cfg



def evaluate(e, cfg, model, val_iter, device):
    model.eval()
    with torch.no_grad():
        loss_list = []
        for b, batch in enumerate(val_iter):

            tgt = batch['rps_feat'].to(device)
            # tgt_gt is NEEDED for safe autoregressive evaluation (to protect GT duration)
            tgt_gt = batch['rps_feat_gt'].to(device)
            rps_mask = batch['rps_mask'].to(device)

            
            out = model(tgt=tgt, tgt_gt=tgt_gt, tgt_key_mask=rps_mask, use_teacher_pos=False)

            loss = model.loss(predict=out, gt=tgt_gt, mask=rps_mask, raw_feats=tgt)

            loss_list.append(loss.data.item())

    return sum(loss_list) / len(loss_list)


def train(e, cfg, model, optimizer, train_iter, device, scheduler=None, scheduler_step=None, grad_clip_norm=None):
    model.train()
    loss_list = []
    
    
    scheduled_sampling_prob = 0.2
    
    for b, batch in enumerate(train_iter):
        # Decide whether to use teacher forcing for position context
        use_teacher_pos = True
        if random.random() < scheduled_sampling_prob:
            use_teacher_pos = False

        # Unpack batch tensors (same as in evaluate) and move to device
        tgt = batch['rps_feat'].to(device)
        # tgt_gt is NEEDED for safe autoregressive evaluation/training (to protect GT duration)
        tgt_gt = batch['rps_feat_gt'].to(device)
        rps_mask = batch['rps_mask'].to(device)

        # Forward
        out = model(tgt=tgt, tgt_gt=tgt_gt, tgt_key_mask=rps_mask, use_teacher_pos=use_teacher_pos)

        # Loss
        optimizer.zero_grad()
        loss, feature_breakdown = model.loss(predict=out,
                             gt=tgt_gt,
                             mask=rps_mask,
                             
                             raw_feats=tgt,
                             return_components=True)
        
        
        feature_breakdown_vals = {}
        for k, v in feature_breakdown.items():
            if hasattr(v, 'item'):
                feature_breakdown_vals[k] = v.item()
            else:
                feature_breakdown_vals[k] = float(v)

        # Backward
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler and scheduler_step == 'batch':
            scheduler.step()

        # Save loss
        loss_list.append(loss.item())

        # print loss
        if b % 100 == 0:
            per_feat_msg = ' '.join([f"{name}:{value:.4f}" for name, value in feature_breakdown_vals.items()])
            print(f"iter({b // 100}) train_loss:{loss.item()} | {per_feat_msg}")
        
        
        del out, loss, feature_breakdown

    return sum(loss_list) / len(loss_list)


def main(args=None):
    args = parse_args() if args is None else args
    if args.resume and args.pretrained:
        raise ValueError("--resume and --pretrained cannot be used together")

    # Config
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        cfg = yaml.full_load(f)
    cfg = sync_rps_feature_dict(cfg, config_path)

    if cfg['use_gpu']:
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
            f.write(f"* Learning Rate:{cfg['lr']}\n")
            f.write(f"* Batch Size:{cfg['batch_size']}\n")
            f.write(f"* GPU:{cfg['gpuID']}\n")
            f.write(f"* Nhead:{cfg['nhead']}\n")
            f.write(f"* Num_layers:{cfg['num_layers']}\n")
    else:
        with open(log_path, 'a') as f:
            f.write(f"# Resume {datetime.datetime.now().isoformat()} from {args.resume}\n")

    # Model
    model = GraphTransformer(cfg=cfg).to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("* Number of parameters: %.2fM *" % (total / 1e6))

    # Optimizer
    base_lr = float(cfg.get('lr', 1e-4))
    weight_decay = float(cfg.get('weight_decay', 0.0))
    optimizer_name = str(cfg.get('optimizer', 'adamw')).lower()
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")

    # Data
    train_dataset = MuGraphDataset(cfg,dataroot=cfg['train_pkl'])
    val_dataset = MuGraphDataset(cfg,dataroot=cfg['val_pkl'])

    num_workers = int(cfg.get('num_workers', 0))
    pin_memory = bool(cfg.get('pin_memory', device.type == 'cuda'))
    persistent_workers = bool(cfg.get('persistent_workers', False)) and num_workers > 0
    prefetch_factor = cfg.get('prefetch_factor', None) if num_workers > 0 else None

    # train DataLoader: keep shuffle and drop_last for stable training batches
    train_dataloader_kwargs = dict(batch_size=cfg['batch_size'],
                                   drop_last=True,
                                   collate_fn=collate_feature,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   persistent_workers=persistent_workers)

    # val DataLoader: do NOT shuffle and do NOT drop_last so validation covers all samples
    val_dataloader_kwargs = dict(batch_size=cfg['batch_size'],
                                 drop_last=False,
                                 collate_fn=collate_feature,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 persistent_workers=persistent_workers)

    if prefetch_factor is not None:
        train_dataloader_kwargs['prefetch_factor'] = prefetch_factor
        val_dataloader_kwargs['prefetch_factor'] = prefetch_factor

    train_dataloader = DataLoader(dataset=train_dataset, **train_dataloader_kwargs)
    val_dataloader = DataLoader(dataset=val_dataset, **val_dataloader_kwargs)

    # Debugging info to help diagnose sudden val loss changes
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    try:
        from math import ceil
        print(f"Train batches: {ceil(len(train_dataset)/cfg['batch_size'])}, Val batches: {ceil(len(val_dataset)/cfg['batch_size'])}")
    except Exception:
        pass

    steps_per_epoch = len(train_dataloader)
    scheduler, scheduler_step = create_scheduler(
        optimizer,
        cfg,
        num_epochs=cfg['num_epochs'],
        steps_per_epoch=max(1, steps_per_epoch)
    )
    grad_clip_norm = float(cfg.get('grad_clip_norm', 0.0) or 0.0)

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
        if scheduler and metadata.get('scheduler_state_dict'):
            try:
                scheduler.load_state_dict(metadata['scheduler_state_dict'])
            except Exception as err:
                print(f"[Scheduler] Failed to load state ({err}); continuing with fresh scheduler.")
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
        train_loss = train(
            epoch,
            cfg,
            model,
            optimizer,
            train_dataloader,
            device,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            grad_clip_norm=grad_clip_norm
        )

        # evaluate
        val_loss = evaluate(e=epoch, cfg=cfg, model=model, val_iter=val_dataloader, device=device)

        # Scheduler update (epoch/val based)
        if scheduler:
            if scheduler_step == 'epoch':
                scheduler.step()
            elif scheduler_step == 'val':
                scheduler.step(val_loss)

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
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'val_best.pt'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_loss': best_loss,
                'config': cfg
            }, os.path.join(model_save_dir, 'checkpoint_val_best.pt'))

        # Save Model Val
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'train_best.pt'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_loss': best_loss,
            'config': cfg
        }, os.path.join(model_save_dir, 'checkpoint_last.pt'))
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'epoch_{epoch}.pt'))

        # log
        with open(log_path, 'a') as f:
            f.write(f'Epoch:{epoch:<5} train_loss:{train_loss:<20} val_loss:{val_loss:<20} best_loss:{best_loss:<20} lr:{current_lr:<20}\n')


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
#     # V = data['rps_feature_seq']
#     # E = data['rps_edge_seq']
#     # out = encoder(V,E)
#     # print(out.shape)
#
#     # # -------------------- [Test Dncoder] --------------------#
#     # decoder = DecoderTransformer(cfg=cfg)
#     # memory = torch.rand((2,128,512))
#     # out = decoder(tgt=data['note_seq'],memory = memory,tgt_mask = data['note_mask'])
#     # print(out,out.shape)
#
#     # -------------------- [Test Graph2Seq] --------------------#
#     model= Graph2Seq(cfg=cfg)
#     out = model(V=data['rps_feature_seq'],E=data['rps_edge_seq'],tgt=data['note_seq'],tgt_mask=data['note_mask'])
#     print(out,out.shape)
#     loss = model.loss(predict=out,gt=data['note_seq_gt'])
#     print(loss)