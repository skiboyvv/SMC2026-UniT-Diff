import code1.data.utils.config as config_module
from code1.data.utils import read_list, maybe_mkdir, test_all_case
from DiffVNet.diff_vnet import DiffVNet
import torch.nn as nn
import torch
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='synapse',
                    help='Task name: la_10, mmwhs_mr2ct, mnms_toA_10')
parser.add_argument('--exp', type=str, default='fully',
                    help='Experiment name')
parser.add_argument('--split', type=str, default='test',
                    help='Split name')
parser.add_argument('--speed', type=int, default=0,
                    help='Speed for sliding window inference')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='GPU ID')
parser.add_argument('--norm', type=str, default='instancenorm',
                    choices=['batchnorm', 'instancenorm'],
                    help='Normalization type used in training')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = config_module.Config(args.task)

# ==========================================================
# Task-Conditioned 推理包装器
# ==========================================================


class TaskCondWrapper(nn.Module):
    """自动注入 task_id 并切片输出"""

    def __init__(self, base_model, task_idx, start_ch, end_ch):
        super().__init__()
        self.base_model = base_model
        self.task_idx = task_idx
        self.start_ch = start_ch
        self.end_ch = end_ch

    def forward(self, image=None, x=None, pred_type=None, step=None):
        # 获取 Batch Size
        if image is not None:
            B = image.shape[0]
        elif x is not None:
            B = x.shape[0]
        else:
            B = 1

        # 动态构造 task_id_tensor
        task_id_tensor = torch.full(
            (B,), self.task_idx, dtype=torch.long, device='cuda')

        # 核心：将 task_id 传入底层网络
        out = self.base_model(
            image=image,
            x=x,
            pred_type=pred_type,
            step=step,
            task_id=task_id_tensor
        )

        # 如果是切片预测类型，截取当前任务需要的通道
        if isinstance(out, torch.Tensor) and out.shape[1] == 11:
            return out[:, self.start_ch:self.end_ch, ...]
        return out


# ==========================================================

if __name__ == '__main__':
    stride_dict = {
        0: (16, 4),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]

    snapshot_path = f'./logs/{args.exp}/'
    test_save_path = f'./logs/{args.exp}/predictions/'
    maybe_mkdir(test_save_path)
    print(f"Testing snapshot path: {snapshot_path}")

    # ==========================================
    # 定义 11 分类的全局映射
    # ==========================================
    GLOBAL_CLASSES = 11
    TASK_CHANNELS = {
        'la_10': (0, 2),
        'mmwhs_mr2ct': (2, 7),
        'mnms_toB_10': (7, 11)
    }
    TASK_ID_MAP = {
        'la_10': 0,
        'mmwhs_mr2ct': 1,
        'mnms_toB_10': 2
    }

    if args.task not in TASK_CHANNELS:
        raise ValueError(f"Task {args.task} not defined in 11-class mapping!")

    current_task_idx = TASK_ID_MAP[args.task]
    start_ch, end_ch = TASK_CHANNELS[args.task]

    # 1. 实例化全局 11 分类的底座模型
    base_model = DiffVNet(
        n_channels=config.num_channels,
        n_classes=GLOBAL_CLASSES,
        n_filters=config.n_filters,
        normalization=args.norm,
        has_dropout=False
    ).cuda()

    # 2. 加载权重
    ckpt_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')

    if not os.path.exists(ckpt_path):
        print(f"❌ ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f">>> Loading Task-Conditioned Checkpoint from: {ckpt_path}")

    with torch.no_grad():
        checkpoint = torch.load(
            ckpt_path, map_location='cuda', weights_only=False)

        # 处理可能的不同key格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        base_model.load_state_dict(state_dict)
        base_model.eval()
        print(f'✅ Successfully loaded Task-Conditioned checkpoint')

        # 3. 构建推理包装器
        model = TaskCondWrapper(
            base_model, current_task_idx, start_ch, end_ch).cuda()
        model.eval()

        # 4. 运行推理
        test_all_case(
            args.task,
            model,
            read_list(args.split, task=args.task),
            num_classes=config.num_cls,
            patch_size=config.patch_size,
            stride_xy=stride[0],
            stride_z=stride[1],
            test_save_path=test_save_path
        )

    print(f"✅ Inference completed! Predictions saved to {test_save_path}")
