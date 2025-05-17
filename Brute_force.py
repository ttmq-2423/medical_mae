import argparse
import datetime
import numpy as np
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, auc
from numpy import interp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ptflops import get_model_complexity_info
import timm

from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import timm.optim.optim_factory as optim_factory

from torchvision import models

import segmentation_models_pytorch as smp

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset_chest_xray
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.sampler import RASampler
from engine_finetune import train_one_epoch, evaluate_chestxray, accuracy

from libauc import losses
from torch.cuda.amp import autocast
# from model.convnext_swin import convnext_base_swin4
# arch_functions = {
#     "convnext_base_swin4": convnext_base_swin4,
#     # "convnext_swin_base_34": convnext_swin_base_34
# }
# from model.conv_vit import ConvNeXtMobileViT
def get_args_parser():
    parser = argparse.ArgumentParser('Brute Force for image classification', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--cutmix', type=float, default=0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')
    parser.add_argument('--finetune', default='')
    parser.add_argument('--global_pool', action='store_true', default=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')
    parser.add_argument('--data_path', default='data/CheXpert-v1.0/', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--vit_dropout_rate', type=float, default=0)
    parser.add_argument('--build_timm_transform', action='store_true', default=False)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--train_list', default=None, type=str)
    parser.add_argument('--val_list', default=None, type=str)
    parser.add_argument('--test_list', default=None, type=str)
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--dataset', default='chexpert', type=str)
    parser.add_argument('--norm_stats', default=None, type=str)
    parser.add_argument('--loss_func', default=None, type=str)
    parser.add_argument('--checkpoint_type', default=None, type=str)
    parser.add_argument('--save', default='results', type=str)
    return parser

def main(args):
    misc.init_distributed_mode(args)
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(f"{args}".replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    os.makedirs(args.save, exist_ok=True)

    dataset_test = build_dataset_chest_xray(split='test', args=args)
    sampler_test = SequentialSampler(dataset_test)
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    if (args.model == "conv_vit"):
        model = ConvNeXtMobileViT(num_classes=5)
        if args.finetune:
                checkpoint = torch.load(args.finetune, map_location=args.device)
                print("Load pre-trained checkpoint from: %s" % args.finetune)
                # print("key: ",checkpoint.keys())
                if 'state_dict' in checkpoint.keys():
                    checkpoint_model = checkpoint['state_dict']
                elif 'model' in checkpoint.keys():
                    checkpoint_model = checkpoint['model']
                else:
                    checkpoint_model = checkpoint
                model.load_state_dict(checkpoint_model, strict=False)
    elif (args.model == "ConvNeXt_Swin"):
            # model = ConvNeXt_Swin(num_classes=5)
            model = arch_functions["convnext_base_swin4"](num_classes=5)
            if args.finetune:
                checkpoint = torch.load(args.finetune, map_location=args.device)
                print("Load pre-trained checkpoint from: %s" % args.finetune)
                # print("key: ",checkpoint.keys())
                if 'state_dict' in checkpoint.keys():
                    checkpoint_model = checkpoint['state_dict']
                elif 'model' in checkpoint.keys():
                    checkpoint_model = checkpoint['model']
                else:
                    checkpoint_model = checkpoint
                model.load_state_dict(checkpoint_model, strict=False)
    elif (args.model == "convnext_base"):
        model = models.__dict__[args.model](num_classes=args.nb_classes)
        checkpoint = torch.load(args.finetune, map_location=args.device)
        if 'state_dict' in checkpoint.keys():
                    checkpoint_model = checkpoint['state_dict']
        elif 'model' in checkpoint.keys():
                    checkpoint_model = checkpoint['model']
        else:
                    checkpoint_model = checkpoint
        model.load_state_dict(checkpoint_model, strict=False)
        
    else: 
        model=  timm.create_model(model_name=args.model,  num_classes=args.nb_classes)
        checkpoint = torch.load(args.finetune, map_location=args.device)
        # print(checkpoint.keys())
        print("Load pre-trained checkpoint from: %s" % args.finetune)
                # print("key: ",checkpoint.keys())
        # print(checkpoint['args'],checkpoint['epoch'])
        if 'state_dict' in checkpoint.keys():
                    checkpoint_model = checkpoint['state_dict']
        elif 'model' in checkpoint.keys():
                    checkpoint_model = checkpoint['model']
                    # print("aaa")
        else:
            checkpoint_model = checkpoint
        # print(checkpoint_model.keys())
        model.load_state_dict(checkpoint_model, strict=False)
        # model=  timm.create_model(model_name=args.model, checkpoint_path=args.finetune, num_classes=args.nb_classes)
        
        # model.load_state_dict(torch.load(args.finetune, weights_only=True))
        # model.load_state_dict(checkpoint_model, strict=False)
        
    model.eval()
    model.to(args.device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_targets, all_outputs = [], []
    input_tensor = torch.randn( 3, 224, 224)
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True)
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}") 
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader_test, 10, header):
            images, target = batch[0].to(device), batch[-1].to(device)
            with autocast():
                output = model(images)
            output = torch.sigmoid(output)
            all_targets.append(target.cpu())
            all_outputs.append(output.cpu())

    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_outputs = torch.cat(all_outputs, dim=0).numpy()

    train_cols = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    class_names = train_cols
    colors = ['lightblue', 'lightgreen', 'peachpuff', 'mediumpurple', 'sandybrown']

    outAUROC, fprs, tprs = [], [], []
    for i in range(5):
        fpr, tpr, _ = roc_curve(all_targets[:, i], all_outputs[:, i])
        auc_i = roc_auc_score(all_targets[:, i], all_outputs[:, i])
        outAUROC.append(round(auc_i, 3))
        fprs.append(fpr)
        tprs.append(tpr)

    print(f"AUC avg: {np.mean(outAUROC) * 100:.2f}%")
    print(f"AUC for each label: {[round(x * 100, 2) for x in outAUROC]}")

    predicted_classes = (all_outputs > 0.5).astype(int)

    # Vẽ tất cả confusion matrix (5 class + tổng thể)
    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.75, hspace=0.55)

    for i in range(5):
        ax = fig.add_subplot(gs[i])
        cm = confusion_matrix(all_targets[:, i], predicted_classes[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'], ax=ax, annot_kws={"size": 9})
        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_ylabel('True', fontsize=7)
        ax.set_title(f'{class_names[i]}', fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=0)

    ax = fig.add_subplot(gs[5])
    cm_total = confusion_matrix(all_targets.flatten(), predicted_classes.flatten())
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'], ax=ax, annot_kws={"size": 7})
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_ylabel('True', fontsize=7)
    ax.set_title('Overall', fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=0)
    cm_filename = f"confusion_matrix_{args.model}.png"
    plt.savefig(os.path.join(args.save, cm_filename), dpi=300, bbox_inches='tight')
    plt.close()

    # Vẽ ROC
    plt.figure(figsize=(7, 5))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    line_styles = ['-', '-', '-', '-', '-']
    line_width = 1.2

    for i in range(5):
        auc_percent = outAUROC[i] * 100
        plt.plot(fprs[i], tprs[i],
                 linestyle=line_styles[i],
                 linewidth=line_width,
                 label=f'{class_names[i]} (AUC = {auc_percent:.2f}%)',
                 color=colors[i],
                 alpha=0.9)

    # Macro-average
    all_fpr = np.unique(np.concatenate([fprs[i] for i in range(5)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += np.interp(all_fpr, fprs[i], tprs[i])
    mean_tpr /= 5
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr,
             color='red',
             linestyle='--',
             linewidth=3.0,
             label=f'Macro-average (AUC = {macro_auc * 100:.2f}%)',
             alpha=0.9)

    # Đường tham chiếu
    plt.plot([0, 1], [0, 1],
             color='#cccccc',
             linestyle='--',
             linewidth=1.0,
             alpha=0.7)

    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves', fontsize=16, fontweight='bold', pad=15)

    plt.legend(loc='lower right', frameon=True,
               fancybox=True, framealpha=0.9, fontsize=10,
               bbox_to_anchor=(1.02, 0.0))

    plt.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=5)

    plt.tight_layout()
    roc_filename = f"ROC_curves_{args.model}_improved.png"
    plt.savefig(os.path.join(args.save, roc_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
