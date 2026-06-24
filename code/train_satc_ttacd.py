# train_satc_ttacd.py

from torchvision import transforms
from data.StrongAug import get_StrongAug, ToTensor, CenterCrop
from data.data_loaders import DatasetAllTasks
from code1.data.utils.config import Config
from code1.data.utils.loss import DC_and_CE_loss, SoftDiceLoss
from code1.data.utils import (EMA, maybe_mkdir, get_lr, fetch_data,
                              GaussianSmoothing, seed_worker,
                              poly_lr, sigmoid_rampup)
from DiffVNet.diff_vnet import DiffVNet
import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--exp',            type=str,   default='satc_ttacd_mnms05')
parser.add_argument('--gpu',            type=str,   default='0')
parser.add_argument('--seed',           type=int,   default=0)
parser.add_argument('--base_lr',        type=float, default=0.01)
parser.add_argument('--num_workers',    type=int,   default=2)
parser.add_argument('--mixed_precision', action='store_true', default=True)
parser.add_argument('--max_epoch',      type=int,   default=300)
parser.add_argument('--sup_loss',       type=str,   default='w_ce+dice')
parser.add_argument('--unsup_loss',     type=str,   default='w_ce+dice')
parser.add_argument('-w', '--mu',       type=float, default=2.0)
parser.add_argument('-s', '--ema_w',    type=float, default=0.99)
parser.add_argument('-r', '--mu_rampup', action='store_true', default=True)
parser.add_argument('--norm',           type=str,   default='instancenorm',
                    choices=['batchnorm', 'instancenorm'])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

GLOBAL_CLASSES = 11
TASK_CHANNELS = {'la_10': (0, 2), 'mmwhs_mr2ct': (
    2, 7), 'mnms_toB_10': (7, 11)}
TASK_KEYS = ['la_10', 'mmwhs_mr2ct', 'mnms_toB_10']

# TTACD：任务类型感知的条件化丢弃率
# SSL(LA)=20%丢弃，UDA(MMWHS)=0%，DG(MNMS)=100%（永不条件化）
TASK_DROPOUT_RATE = {
    'la_10':       0.8,     
    'mmwhs_mr2ct': 0.0,
    'mnms_toB_10': 0.5
}                              #la0.2

# 对应的任务 ID（丢弃后传 None）
TASK_ID_BASE = {
    'la_10':       0,
    'mmwhs_mr2ct': 1,
    'mnms_toB_10': 2   
}

TASK_LOSS_WEIGHT = {
    'la_10':       1.5,   # 加大 LA 的优化权重
    'mmwhs_mr2ct': 1.0,
    'mnms_toB_10': 1.0
}

def get_task_id_tensor(task_key, batch_size, device):
    """
    TTACD：根据任务类型和丢弃率，决定是否注入任务 token。
    返回 task_id_tensor 或 None。
    """
    dropout_rate = TASK_DROPOUT_RATE[task_key]
    if random.random() < dropout_rate:
        return None   # 本次不注入：强化中性路径
    else:
        return torch.full(
            (batch_size,), TASK_ID_BASE[task_key],
            dtype=torch.long, device=device)


def make_loaders_equivalent(stage_info, config):
    task = stage_info['task']
    transforms_train = get_StrongAug(config.patch_size, 3, 0.7)
    ds_l_raw = DatasetAllTasks(split=stage_info['sl'], task=task,
                               num_cls=config.num_cls, transform=transforms_train)
    ds_u_raw = DatasetAllTasks(split=stage_info['su'], task=task,
                               num_cls=config.num_cls, transform=transforms_train,
                               unlabeled=stage_info['unlabeled_flag'])
    if "mmwhs" not in task:
        ds_l = DatasetAllTasks(split=stage_info['sl'], task=task,
                               num_cls=config.num_cls, transform=transforms_train,
                               repeat=len(ds_u_raw.ids_list))
        ds_u = ds_u_raw
    else:
        ds_l = ds_l_raw
        ds_u = DatasetAllTasks(split=stage_info['su'], task=task,
                               num_cls=config.num_cls, transform=transforms_train,
                               unlabeled=stage_info['unlabeled_flag'],
                               repeat=len(ds_l_raw.ids_list))
    loader_l = DataLoader(ds_l, batch_size=config.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          worker_init_fn=seed_worker, drop_last=True)
    loader_u = DataLoader(ds_u, batch_size=config.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          worker_init_fn=seed_worker, drop_last=True)
    ds_eval = DatasetAllTasks(split=stage_info['se'], is_val=True, task=task,
                              num_cls=config.num_cls,
                              transform=transforms.Compose(
                                  [CenterCrop(config.patch_size), ToTensor()]))
    return loader_l, loader_u, DataLoader(ds_eval, pin_memory=True)


def make_loss_function(name, weight=None):
    if name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    return DC_and_CE_loss()


class Difficulty:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def cal_weights(self, pred, label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        mom = (self.accumulate_iters - 1) / self.accumulate_iters
        cur_cls_learn = torch.where(
            delta_dice > 0, delta_dice,
            torch.zeros_like(delta_dice)) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(
            delta_dice <= 0, delta_dice,
            torch.zeros_like(delta_dice)) * torch.log(cur_dice / self.last_dice)
        self.last_dice = cur_dice
        self.cls_learn = EMA(cur_cls_learn,   self.cls_learn,   momentum=mom)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=mom)
        cur_diff = torch.pow(
            (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8), 1/5)
        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=mom)
        return (cur_diff * self.dice_weight) * self.num_cls


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    snapshot_path = f'./logs/{args.exp}_{args.norm}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S', force=True)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    task_infos = {
        'la_10':       {'task': 'la_10',       'sl': 'labeled_0.1',
                        'su': 'unlabeled_0.1', 'se': 'eval_0.1',
                        'unlabeled_flag': True},
        'mmwhs_mr2ct': {'task': 'mmwhs_mr2ct', 'sl': 'train_mr2ct_labeled',
                        'su': 'train_mr2ct_unlabeled', 'se': 'eval_mr2ct',
                        'unlabeled_flag': True},
        'mnms_toB_10': {'task': 'mnms_toB_10', 'sl': 'train_toB_labeled_0.05',
                        'su': 'train_toB_unlabeled_0.05', 'se': 'eval_toB_0.05',
                        'unlabeled_flag': True}
    }

    configs, loaders, diff_calculators, task_losses = {}, {}, {}, {}
    best_evals = {t: 0.0 for t in TASK_KEYS}

    for t_key in TASK_KEYS:
        configs[t_key] = Config(t_key)
        loader_l, loader_u, loader_e = make_loaders_equivalent(
            task_infos[t_key], configs[t_key])
        loaders[t_key] = {'l': loader_l, 'u': loader_u, 'e': loader_e}
        diff_calculators[t_key] = Difficulty(
            configs[t_key].num_cls, accumulate_iters=50)
        task_losses[t_key] = {
            'sup':   make_loss_function(args.sup_loss),
            'unsup': make_loss_function(args.unsup_loss),
            'deno':  make_loss_function(args.sup_loss)
        }

    model = DiffVNet(
        n_channels=configs['la_10'].num_channels,
        n_classes=GLOBAL_CLASSES,
        n_filters=configs['la_10'].n_filters,
        normalization=args.norm, has_dropout=True
    ).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=3e-5, nesterov=True)
    amp_grad_scaler = GradScaler() if args.mixed_precision else None

    for epoch_num in range(args.max_epoch):
        model.train()
        iters = {t: {'l': iter(loaders[t]['l']),
                     'u': iter(loaders[t]['u'])} for t in TASK_KEYS}

        dataset_pool = []
        for t in TASK_KEYS:
            dataset_pool.extend([t] * len(loaders[t]['l']))
        random.shuffle(dataset_pool)

        epoch_losses = {t: [] for t in TASK_KEYS}
        optimizer.param_groups[0]['lr'] = poly_lr(
            epoch_num, args.max_epoch, args.base_lr, 0.9)
        mu = (args.mu * sigmoid_rampup(epoch_num, args.max_epoch)
              if args.mu_rampup else args.mu)

        pbar = tqdm(dataset_pool, desc=f"Epoch {epoch_num}")

        for task_key in pbar:
            batch_l = next(iters[task_key]['l'])
            batch_u = next(iters[task_key]['u'])
            start_ch, end_ch = TASK_CHANNELS[task_key]

            # ── EMA────────────────────────
            for name, p_theta in model.decoder_theta.named_parameters():
                sd_xi = model.denoise_model.decoder.state_dict()
                sd_psi = model.decoder_psi.state_dict()
                if name in sd_xi and p_theta.shape == sd_xi[name].shape:
                    p_xi = sd_xi[name]
                    p_psi = sd_psi[name]
                    with torch.no_grad():
                        if 'out_conv' in name:
                            p_theta.data[start_ch:end_ch] = (
                                args.ema_w * p_theta.data[start_ch:end_ch]
                                + (1 - args.ema_w)
                                * (p_xi.data[start_ch:end_ch]
                                   + p_psi.data[start_ch:end_ch]) / 2.0)
                        else:
                            p_theta.data = (
                                args.ema_w * p_theta.data
                                + (1 - args.ema_w)
                                * (p_xi.data + p_psi.data) / 2.0)

            optimizer.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            label_l = label_l.long()
            image_u = fetch_data(batch_u, labeled=False)
            cur_conf = configs[task_key]

            # ── TTACD：任务类型感知的条件化丢弃 ─────────────────────
            task_id_tensor = get_task_id_tensor(
                task_key, cur_conf.batch_size, 'cuda')
            # MNMS dropout=1.0 → task_id_tensor 永远是 None
            # LA dropout=0.2 → 80%概率有 task_id，20%概率为 None
            # MMWHS dropout=0.0 → 始终有 task_id

            with autocast(enabled=args.mixed_precision):
                shp = (cur_conf.batch_size, GLOBAL_CLASSES) + \
                    cur_conf.patch_size
                label_l_onehot = torch.zeros(shp).cuda()
                label_l_onehot.scatter_(1, label_l + start_ch, 1)
                x_start = label_l_onehot * 2 - 1

                # ── D_xi 路径（去噪监督）────────────────────────────
                x_t, t, noise = model(x=x_start, pred_type="q_sample")
                # SATC：从 alphas_cumprod 取当前时间步的值，传给 forward
                # DiffVNet.forward 会把它转给 TimeStepEmbedding
                p_l_xi_11 = model(x=x_t, step=t, image=image_l,
                                  pred_type="D_xi_l",
                                  task_id=task_id_tensor)

                # ── D_psi 路径（直接分割监督）────────────────────────
                p_l_psi_11 = model(image=image_l, pred_type="D_psi_l",
                                   task_id=task_id_tensor)

                p_l_xi = p_l_xi_11[:, start_ch:end_ch, ...]
                p_l_psi = p_l_psi_11[:, start_ch:end_ch, ...]

                L_deno = task_losses[task_key]['deno'](p_l_xi, label_l)
                weight_diff = diff_calculators[task_key].cal_weights(
                    p_l_xi.detach(), label_l)
                task_losses[task_key]['sup'].update_weight(weight_diff)
                L_diff = task_losses[task_key]['sup'](p_l_psi, label_l)

                # ── 无监督分支 ──────────────────────────────────────
                with torch.no_grad():
                    p_u_xi_11 = model(image_u, pred_type="ddim_sample",
                                      task_id=task_id_tensor)
                    p_u_psi_11 = model(image_u, pred_type="D_psi_l",
                                       task_id=task_id_tensor)
                    p_u_xi = p_u_xi_11[:, start_ch:end_ch, ...]
                    p_u_psi = p_u_psi_11[:, start_ch:end_ch, ...]
                    smoothing = GaussianSmoothing(cur_conf.num_cls, 3, 1)
                    p_u_xi = smoothing(F.gumbel_softmax(p_u_xi, dim=1))
                    p_u_psi = F.softmax(p_u_psi, dim=1)
                    pseudo_label = torch.argmax(
                        p_u_xi + p_u_psi, dim=1, keepdim=True)

                p_u_theta_11 = model(image=image_u, pred_type="D_theta_u",
                                     task_id=task_id_tensor)
                p_u_theta = p_u_theta_11[:, start_ch:end_ch, ...]
                L_u = task_losses[task_key]['unsup'](
                    p_u_theta, pseudo_label.detach())

                w = TASK_LOSS_WEIGHT[task_key]
                loss = w * (L_deno + L_diff + mu * L_u)

            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            epoch_losses[task_key].append(loss.item())

        # ── 验证与保存──────────────────────────
        if epoch_num % 5 == 0 or epoch_num == args.max_epoch - 1:
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
            val_scores = {}

            for t_key in TASK_KEYS:
                c_conf = configs[t_key]
                s_ch, e_ch = TASK_CHANNELS[t_key]
                dice_list = [[] for _ in range(c_conf.num_cls - 1)]

                for batch in loaders[t_key]['e']:
                    with torch.no_grad():
                        image, gt = fetch_data(batch)
                        # 验证时：与训练时的"默认行为"一致
                        # MNMS 始终 None；LA/MMWHS 使用固定 task_id（不做随机丢弃）
                        if TASK_DROPOUT_RATE[t_key] < 1.0:
                            val_task_id = torch.full(
                                (image.shape[0],), TASK_ID_BASE[t_key],
                                dtype=torch.long, device='cuda')
                        else:
                            val_task_id = None

                        p11 = model(image, pred_type="D_theta_u",
                                    task_id=val_task_id)
                        p_out = p11[:, s_ch:e_ch, ...]

                        shp = (p_out.shape[0],
                               c_conf.num_cls) + p_out.shape[2:]
                        y_oh = torch.zeros(shp).cuda()
                        y_oh.scatter_(1, gt.long(), 1)
                        x_oh = torch.zeros(shp).cuda()
                        pred = torch.argmax(p_out, dim=1, keepdim=True).long()
                        x_oh.scatter_(1, pred, 1)
                        dice = dice_func(x_oh, y_oh,
                                         is_training=False).data.cpu().numpy()
                        for idx, d in enumerate(dice):
                            dice_list[idx].append(d)

                mean_dice = float(np.mean([np.mean(dl)
                                           for dl in dice_list if dl]))
                val_scores[t_key] = mean_dice
                writer.add_scalar(f'val_dice/{t_key}', mean_dice, epoch_num)

                if mean_dice > best_evals[t_key]:
                    best_evals[t_key] = mean_dice
                    save_path = os.path.join(
                        snapshot_path, f'ckpts/best_model_{t_key}.pth')
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    logging.info(
                        f"[{t_key}] New Best Dice: {mean_dice:.4f} @ Epoch {epoch_num}")

            avg = np.mean(list(val_scores.values()))
            logging.info(
                f"Epoch {epoch_num} | "
                + " | ".join(f"{k}: {v:.4f}" for k, v in val_scores.items())
                + f" | Avg: {avg:.4f}")

        total_loss = np.mean([l for t in TASK_KEYS for l in epoch_losses[t]])
        writer.add_scalar('loss/total', total_loss, epoch_num)
        writer.add_scalar('lr', get_lr(optimizer), epoch_num)

    writer.close()
    logging.info("Training complete.")
