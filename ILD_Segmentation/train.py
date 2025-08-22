#region 模块1：依赖导入和配置
# 功能：导入必要的库和模块，设置基础配置
import os
import glob
import random
import monai
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder
import cv2
from matplotlib import pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
import swanlab
import json
import pytz
#endregion

#region 模块2：命令行参数配置
# 功能：定义训练脚本的所有命令行参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--tr_npy_path',
    type=str,
    default='/home/huchengpeng/workspace/MedSAM-main/ILD/data',
    help="Path to the data root directory.",
)
parser.add_argument(
    '-medsam_checkpoint',
    type=str,
    default='/home/huchengpeng/workspace/MedSAM-main/ILD/checkpoints/medsam_vit_b.pth',
    help="Path to the MedSAM checkpoint.",
)
parser.add_argument(
    '-work_dir',
    type=str,
    default="/home/huchengpeng/workspace/MedSAM-main/ILD/workdir",
    help="Path to where the checkpoints and logs are saved."
)
parser.add_argument(
    '-max_epochs',
    type=int,
    default=200,  # 增加epoch数，适应小数据集的充分训练
    help="Maximum number of epochs."
)
parser.add_argument(
    '-batch_size',
    type=int,
    default=8,  # 减小批次大小，适应2000张切片的小数据集
    help="Batch size."
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=8,
    help="Number of data loader workers."
)
parser.add_argument(
    '-resume',
    type=str,
    default=None,
    help="Path to the checkpoint to resume from."
)
parser.add_argument(
    '-lr',
    type=float,
    default=0.0001,  # 提高学习率，加快小数据集的收敛
    help="learning rate (absolute lr)"
)
parser.add_argument(
    '-weight_decay',
    type=float,
    default=0.01,
    help="Weight decay."
)
parser.add_argument(
    '-seed',
    type=int,
    default=42,
    help="Random seed for reproducibility."
)
parser.add_argument(
    '--disable_aug',
    action='store_true',
    help="Disable data augmentation."
)
parser.add_argument(
    '--early_stopping',
    action='store_true',
    help="Enable early stopping."
)
parser.add_argument(
    '--early_stopping_patience',
    type=int,
    default=5,
    help="Patience for early stopping."
)
parser.add_argument(
    '--early_stopping_min_delta',
    type=float,
    default=0.001,
    help="Minimum change in monitored quantity to qualify as an improvement."
)
parser.add_argument(
    '--warmup_epochs',
    type=int,
    default=10,  # warmup epoch数，适应小数据集的稳定训练
    help="Number of warmup epochs for learning rate."
)
parser.add_argument(
    '--negative_sampling_rate',
    type=float,
    default=0.25,
    help="Rate for negative sampling, i.e., training with an absent label."
)
#endregion

#region 模块3：工具函数和辅助类
# 功能：包含早停机制、实验名称生成等辅助功能

class EarlyStopping:
    """早停机制类"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """检查是否应该早停
        
        Args:
            val_loss (float): 当前验证损失
            model: 模型对象
            
        Returns:
            bool: 是否应该早停
        """
        # 处理边界条件：首次调用或验证损失为无效值
        if self.best_loss is None or not np.isfinite(val_loss):
            if np.isfinite(val_loss):  # 只有当验证损失有效时才更新
                self.best_loss = val_loss
                self.save_checkpoint(model)
            return False
            
        # 检查是否有改善
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        # 检查是否达到早停条件
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        try:
            self.best_weights = deepcopy(model.state_dict())
        except Exception as e:
            print(f"Warning: Failed to save model weights for early stopping: {e}")
            self.best_weights = None

def generate_experiment_name(num_epochs, batch_size, lr, weight_decay, seed, data_aug):
    """
    根据超参数和北京时间生成实验名称
    """
    import pytz
    from datetime import datetime
    
    # 获取北京时间
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.now(beijing_tz)
    time_str = beijing_time.strftime("%Y%m%d_%H%M%S")
    
    # 生成实验名称
    aug_str = "aug" if data_aug else "noaug"
    exp_name = f"MedSAM_ILD_ep{num_epochs}_bs{batch_size}_lr{lr:.0e}_{time_str}"
    
    return exp_name

args = parser.parse_args()

# 解析训练参数
data_root = args.tr_npy_path
num_epochs = args.max_epochs
batch_size = args.batch_size
num_workers = args.num_workers
medsam_checkpoint = args.medsam_checkpoint
data_aug = not args.disable_aug
negative_sampling_rate = args.negative_sampling_rate
seed = args.seed
device = "cuda:3"

# 生成实验名称
exp_name = generate_experiment_name(num_epochs, batch_size, args.lr, args.weight_decay, seed, data_aug)
print(f"Experiment name: {exp_name}")

# 创建带实验名称的工作目录
work_dir = join(args.work_dir, exp_name)
makedirs(work_dir, exist_ok=True)
print(f"Work directory: {work_dir}")

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 设置随机种子以确保实验可重现性
def set_random_seeds(seed):
    """设置所有相关库的随机种子"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        # 确保CUDA操作的确定性（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seeds(seed)
print(f"Random seed set to: {seed}")
#endregion

#region 模块4：数据处理
# 功能：数据集类定义，负责数据加载、预处理和增强

class NpyDataset(Dataset):
    def __init__(self, data_root, image_size=1024, data_aug=True, negative_sampling_rate=0.0):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
        self.image_size = image_size
        self.data_aug = data_aug
        self.negative_sampling_rate = negative_sampling_rate
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   
        self.label_dict = {
            1: ["normal", "healthy lung"],
            2: ["ground glass opacity", "GGO"],
            3: ["reticulation", "reticular pattern"],
            4: ["consolidation", "airspace consolidation"],
            5: ["honeycombing", "honeycomb pattern"],
        }
        self.all_lesion_label_ids = [2, 3, 4, 5]
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
        img_1024 = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_1024)<=1.0 and np.min(img_1024)>=0.0, 'image should be normalized to [0, 1]'
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        if gt.shape[0] != 256 or gt.shape[1] != 256:
            ## To match the shape of low_res_masks
            gt_resize = cv2.resize(
                gt,
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            gt_resize = gt.astype(np.uint8)

        present_label_ids = np.unique(gt_resize)
        # Exclude background label 0
        present_lesion_label_ids = [label for label in present_label_ids if label > 1]

        # Negative sampling: train with an absent label
        if random.random() < self.negative_sampling_rate:
            absent_lesion_label_ids = [label for label in self.all_lesion_label_ids if label not in present_lesion_label_ids]
            if absent_lesion_label_ids:
                label_id = random.choice(absent_lesion_label_ids)
                gt2D = np.zeros_like(gt_resize, dtype=np.uint8)
            else:
                # Fallback to positive sampling if all lesions are present
                label_id = random.choice(present_lesion_label_ids)
                gt2D = np.uint8(gt_resize == label_id)
        else:
            # Positive sampling: train with a present label
            # Prioritize lesion labels over normal label
            if present_lesion_label_ids:
                label_id = random.choice(present_lesion_label_ids)
            else:
                # This case should ideally not happen if dataset only contains lesion slices for positive sampling
                label_id = 1 # Fallback to normal
            gt2D = np.uint8(gt_resize == label_id)

        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
        gt2D = np.uint8(gt2D > 0)

        ## Ramdonly select a synonum of the label
        caption = random.choice(self.label_dict[label_id])
        text_token = self.tokenize_text(caption)

        return {
            "image": torch.tensor(img_1024).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "text": [caption],
            "token": text_token,
            "image_name": img_name
        }

    def tokenize_text(self, text):
        """
        Tokenize text using CLIP tokenizer
        """
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
        ).input_ids.squeeze(0)

# Text Prompt Encoder class
class TextPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim = 256,
        image_embedding_size = (64, 64),
        input_image_size = (1024, 1024),
        mask_in_chans = 1,
        activation = nn.GELU,
        ) -> None:
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        text_encoder.requires_grad_(False)
        self.text_encoder = text_encoder
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(
        self, points,
        boxes,
        masks,
        tokens,
    ):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          tokens (torch.Tensor or none): text tokens to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, tokens)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if tokens is not None:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
    
    def _get_batch_size(self, points, boxes, masks, tokens):
        """
        Returns the batch size of the inputs.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif tokens is not None:
            return tokens.shape[0]
        else:
            return 1

# MedSAM model class
class MedSAM(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                freeze_image_encoder=True,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder except for text_encoder_head
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.text_encoder_head.parameters():
            param.requires_grad = True
        
        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, image, tokens):
        # do not compute gradients for image encoder
        with torch.no_grad():
            image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            tokens=tokens,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

# 多标签Dice计算函数
def compute_dice_coefficient(mask_gt, mask_pred):
    """计算Soerensen-Dice系数"""
    intersection = np.sum(mask_gt * mask_pred)
    return 2 * intersection / (np.sum(mask_gt) + np.sum(mask_pred))

# 评估数据集类（用于多标签评估）
class EvalDataset(Dataset):
    def __init__(self, data_root, image_size=1024):
        self.data_root = data_root
        self.image_size = image_size
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "**", "*.npy"), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
        
        # 定义病灶类别名称
        self.lesion_names = {
            1: ["normal", "healthy lung"],
            2: ["ground glass opacity", "GGO"],
            3: ["reticulation", "reticular pattern"],
            4: ["consolidation", "airspace consolidation"],
            5: ["honeycombing", "honeycomb pattern"],
        }
        
        # 初始化tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
    def __len__(self):
        return len(self.gt_path_files)
        
    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        img_1024 = np.transpose(img_1024, (2, 0, 1)) # (3, H, W)
        assert img_1024.shape == (3, self.image_size, self.image_size), f"image shape is {img_1024.shape}, expected ({3, self.image_size, self.image_size})"
        
        gt_data = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels (H, W)
        assert gt_data.shape == (self.image_size, self.image_size), f"ground truth shape is {gt_data.shape}, expected ({self.image_size, self.image_size})"
        
        return {
            "image": torch.tensor(img_1024).float(),
            "gt_multi": torch.tensor(gt_data).long(),  # 多标签ground truth
            "img_name": img_name
        }
    
    def tokenize_text(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        return tokens["input_ids"].squeeze(0)

# 统一验证函数
def validate_model_unified(model, valid_loader, eval_loader, seg_loss, ce_loss, device):
    """
    统一的验证函数，计算验证损失和每类Dice指标
    Args:
        model: 模型
        valid_loader: 单标签验证数据加载器
        eval_loader: 多标签评估数据加载器
        seg_loss: 分割损失函数
        ce_loss: 交叉熵损失函数
        device: 设备
    Returns:
        avg_val_loss: 平均验证损失
        avg_dice: 平均单标签Dice分数
        multiclass_dice_scores: 每类Dice分数字典
    """
    model.eval()
    val_losses = []
    dice_scores = []
    all_dice_scores = []
    
    with torch.no_grad():
        # 计算验证损失和单标签Dice
        for batch in tqdm(valid_loader, desc="Validating Loss & Dice"):
            image, gt2D = batch["image"].to(device), batch["gt2D"].to(device)
            tokens = batch["token"].to(device)
            
            pred = model(image, tokens)
            loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
            val_losses.append(loss.item())
            
            # 计算单标签Dice分数
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            
            for i in range(pred_binary.shape[0]):
                pred_i = pred_binary[i].flatten()
                gt_i = gt2D[i].float().flatten()
                
                intersection = (pred_i * gt_i).sum()
                dice = (2. * intersection) / (pred_i.sum() + gt_i.sum() + 1e-8)
                dice_scores.append(dice.item())
        
        # 计算多类别Dice - 对每个类别独立进行预测和评估
        class_predictions = {1: [], 2: [], 3: [], 4: [], 5: []}  # 存储每个类别的预测结果
        gt_data_list = []  # 存储所有GT数据
        
        for batch in tqdm(eval_loader, desc="Multi-class Validation"):
            image = batch["image"].to(device)
            gt_multi = batch["gt_multi"].cpu().numpy()  # (B, H, W)
            
            for batch_idx in range(image.shape[0]):
                single_image = image[batch_idx:batch_idx+1]  # (1, 3, H, W)
                single_gt = gt_multi[batch_idx]  # (H, W)
                gt_data_list.append(single_gt)
                
                # 对每个类别独立进行预测
                for class_id in [1, 2, 3, 4, 5]:  # 对所有类别进行预测
                    class_names = eval_loader.dataset.lesion_names[class_id]
                    selected_name = np.random.choice(class_names)
                    
                    tokens = eval_loader.dataset.tokenize_text(selected_name).unsqueeze(0).to(device)
                    
                    pred = model(single_image, tokens)
                    pred_sigmoid = torch.sigmoid(pred)
                    pred_binary = (pred_sigmoid > 0.5).cpu().numpy()[0, 0]  # (256, 256)
                    
                    # 将预测结果从256x256缩放到1024x1024
                    pred_binary_resized = cv2.resize(
                        pred_binary.astype(np.uint8),
                        (1024, 1024),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    class_predictions[class_id].append(pred_binary_resized)
        
        # 计算每个类别的Dice分数
        for i, gt_data in enumerate(gt_data_list):
            dice_scores_multi = {}
            label_dict = {1: "normal", 2: "GGO", 3: "reticulation", 4: "consolidation", 5: "honeycombing"}
            
            for class_id in [1, 2, 3, 4, 5]:
                gt_class = (gt_data == class_id).astype(np.uint8)
                pred_class = class_predictions[class_id][i].astype(np.uint8)
                
                if np.sum(gt_class) == 0 and np.sum(pred_class) == 0:
                    dice_scores_multi[label_dict[class_id]] = np.nan
                elif np.sum(gt_class) == 0 and np.sum(pred_class) > 0:
                    dice_scores_multi[label_dict[class_id]] = 0.0
                else:
                    dice_scores_multi[label_dict[class_id]] = compute_dice_coefficient(gt_class, pred_class)
            
            all_dice_scores.append(dice_scores_multi)
    
    model.train()
    
    # 计算平均值
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_dice = sum(dice_scores) / len(dice_scores)
    
    # 计算多类别平均Dice分数
    multiclass_dice_scores = {}
    if len(all_dice_scores) > 0:
        for key in all_dice_scores[0].keys():
            scores = [score[key] for score in all_dice_scores if not np.isnan(score[key])]
            multiclass_dice_scores[key] = np.mean(scores) if len(scores) > 0 else 0.0
    
    return avg_val_loss, avg_dice, multiclass_dice_scores
#endregion

#region 模块5：模型定义和初始化
# 功能：加载预训练模型，构建文本提示编码器，设置训练模式

# 加载SAM预训练模型
print(f"Loading SAM model with checkpoint: {medsam_checkpoint}")
if not os.path.exists(medsam_checkpoint):
    raise FileNotFoundError(f"Model checkpoint not found: {medsam_checkpoint}")

sam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)

# 创建文本提示编码器
text_prompt_encoder = TextPromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size = (1024, 1024),
    mask_in_chans=1,
    activation=nn.GELU,
)

# 加载MedSAM预训练的提示编码器权重（除了文本编码器部分）
medsam_prompt_encoder_state_dict = sam_model.prompt_encoder.state_dict()
for keys in text_prompt_encoder.state_dict().keys():
    if keys in medsam_prompt_encoder_state_dict.keys():
        text_prompt_encoder.state_dict()[keys] = deepcopy(medsam_prompt_encoder_state_dict[keys])
    else:
        assert keys.startswith("text_encoder")
# 计算文本提示编码器参数
text_encoder_params = sum(p.numel() for p in text_prompt_encoder.parameters())
text_encoder_size_mb = text_encoder_params * 4 / (1024 * 1024)  # 假设float32，每个参数4字节
print(f"Text Prompt Encoder size: {text_encoder_params:,} parameters ({text_encoder_size_mb:.2f} MB)")

# 构建完整的MedSAM模型
medsam_model = MedSAM(
    image_encoder = sam_model.image_encoder,
    mask_decoder = deepcopy(sam_model.mask_decoder),
    prompt_encoder = text_prompt_encoder,
    freeze_image_encoder = True
)
medsam_model = medsam_model.to(device)
medsam_model.train()

# 详细的模型参数分析
total_params = sum(p.numel() for p in medsam_model.parameters())
total_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32，每个参数4字节
trainable_params = sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
trainable_size_mb = trainable_params * 4 / (1024 * 1024)
frozen_params = total_params - trainable_params
frozen_size_mb = frozen_params * 4 / (1024 * 1024)

print("\n=== 模型参数分析 ===")
print(f"总参数数量: {total_params:,} ({total_size_mb:.2f} MB)")
print(f"可训练参数: {trainable_params:,} ({trainable_size_mb:.2f} MB)")
print(f"冻结参数: {frozen_params:,} ({frozen_size_mb:.2f} MB)")
print(f"可训练参数占比: {trainable_params/total_params*100:.2f}%")

# 各组件参数分析
image_encoder_params = sum(p.numel() for p in medsam_model.image_encoder.parameters())
mask_decoder_params = sum(p.numel() for p in medsam_model.mask_decoder.parameters())
prompt_encoder_params = sum(p.numel() for p in medsam_model.prompt_encoder.parameters())

print("\n=== 各组件参数分布 ===")
print(f"图像编码器: {image_encoder_params:,} ({image_encoder_params*4/(1024*1024):.2f} MB)")
print(f"掩码解码器: {mask_decoder_params:,} ({mask_decoder_params*4/(1024*1024):.2f} MB)")
print(f"提示编码器: {prompt_encoder_params:,} ({prompt_encoder_params*4/(1024*1024):.2f} MB)")

optim_params = list(
        medsam_model.prompt_encoder.text_encoder_head.parameters()
    ) + list(
        medsam_model.mask_decoder.parameters()
    )
optimizer = optim.AdamW(
    optim_params,
    lr = args.lr,
    betas = (0.9, 0.999),
    eps = 1e-08,
    weight_decay = args.weight_decay
)
print(f"\n优化器参数数量: {sum(p.numel() for p in optim_params):,} ({sum(p.numel() for p in optim_params)*4/(1024*1024):.2f} MB)")



# 初始化早停机制
early_stopping = None
if args.early_stopping:
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        restore_best_weights=True
    )
    print(f"Early stopping enabled with patience: {args.early_stopping_patience}")

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
#endregion

#region 模块6：数据加载器设置
# 功能：创建训练、验证和测试数据加载器

# 创建训练和验证数据集
print("Creating datasets...")
train_dataset = NpyDataset(data_root=join(data_root, 'train'), data_aug=data_aug, negative_sampling_rate=negative_sampling_rate)
valid_dataset = NpyDataset(data_root=join(data_root, 'valid'), data_aug=False, negative_sampling_rate=negative_sampling_rate)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# 创建多标签评估数据集
valid_eval_dataset = EvalDataset(data_root=join(data_root, 'valid'))
test_eval_dataset = EvalDataset(data_root=join(data_root, 'test'))
valid_eval_loader = DataLoader(valid_eval_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
test_eval_loader = DataLoader(test_eval_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")
print(f"Valid eval dataset size: {len(valid_eval_dataset)}")
print(f"Test eval dataset size: {len(test_eval_dataset)}")

# 计算总训练步数和warmup步数
total_steps = len(train_loader) * num_epochs
warmup_steps = args.warmup_epochs * len(train_loader)
min_lr = args.lr * 0.01  # 最小学习率为初始学习率的1%

# 初始化余弦退火+warmup学习率调度器
scheduler = None
if args.warmup_epochs > 0:
    # 创建线性Warmup调度器
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # 从1%的学习率开始
        end_factor=1.0,     # 到达100%学习率
        total_iters=warmup_steps
    )
    
    # 创建余弦退火调度器
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps,
        eta_min=min_lr
    )
    
    # 组合warmup和余弦退火调度器
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
else:
    # 没有warmup，直接使用余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=min_lr
    )

print(f"Learning rate scheduler: CosineAnnealingLR with warmup={args.warmup_epochs > 0}")
print(f"Total training steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")

# 初始化SwanLab
swanlab.init(
    project="MedSAM-ILD-Segmentation",
    experiment_name=exp_name,
    description=f"MedSAM text prompt training for ILD segmentation",
    config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": seed,
        "data_augmentation": data_aug,
        "device": device,
        "medsam_checkpoint": medsam_checkpoint,
        "data_root": data_root,
        "work_dir": work_dir,
        "lr_scheduler": "CosineAnnealingLR_with_warmup",
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "warmup_epochs": args.warmup_epochs,
        "dataset_size": "~2000_slices",
        "optimization_target": "small_dataset_training"
    }
)

resume = args.resume
if resume:
    checkpoint = torch.load(resume, weights_only=False)
    medsam_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    
    # 恢复学习率调度器状态
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Loaded scheduler state from checkpoint")
    
    print(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}")
else:
    start_epoch = 0
    best_loss = None  # 初始化为None，避免记录极大值
#endregion

#region 模块7：训练循环
# 功能：执行模型训练、验证和性能监控

# 初始化训练记录
train_losses = []
val_losses = []
val_dice_scores = []
epoch_time = []

# 初始化全局step计数器
global_step = 0
if resume and "global_step" in checkpoint:
    global_step = checkpoint["global_step"]

print("\nStarting training...")
print(f"Training from epoch {start_epoch} to {num_epochs-1}")
print(f"Total epochs to train: {num_epochs - start_epoch}")
print(f"Starting from global step: {global_step}")
print(f"Total training steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")
print(f"Using scheduler: {type(scheduler).__name__ if scheduler else 'None'}")

for epoch in range(start_epoch, num_epochs):
    epoch_loss = [1e10 for _ in range(len(train_loader))]
    epoch_start_time = time()
    
    # 训练阶段
    medsam_model.train()
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        try:
            # 清零梯度
            optimizer.zero_grad()
            
            # 数据预处理：将数据移动到指定设备
            image, gt2D = batch["image"].to(device), batch["gt2D"].to(device)
            tokens = batch["token"].to(device)
            
            # 前向传播：通过模型获取预测结果
            medsam_pred = medsam_model(image, tokens)
            
            # 损失计算：结合Dice损失和交叉熵损失
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            epoch_loss[step] = loss.item()
            
            # 反向传播：计算梯度
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(medsam_model.parameters(), max_norm=1.0)
            
            # 参数更新：应用梯度
            optimizer.step()
            
            # 更新学习率调度器
            if scheduler is not None:
                scheduler.step()
            
            # 更新全局step计数器
            global_step += 1
            
            # 每个step记录训练指标到SwanLab
            step_log_dict = {
                "step": global_step,
                "Training/loss_step": epoch_loss[step],
                "Training/learning_rate_step": optimizer.param_groups[0]['lr'],
                "Training/warmup_progress": min(1.0, global_step / warmup_steps) if warmup_steps > 0 else 1.0
            }
            swanlab.log(step_log_dict)
            
            # 更新进度条显示
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"Epoch {epoch}, Step {global_step}, loss: {epoch_loss[step]:.4f}, lr: {current_lr:.6f}")
            
        except Exception as e:
            print(f"Error in training step {step}: {str(e)}")
            # 跳过这个batch，继续训练
            epoch_loss[step] = float('inf')
            continue

    # 验证阶段 - 使用统一验证函数评估模型性能
    # 计算验证损失、单标签Dice分数和多类别Dice分数
    val_loss, val_dice, val_multiclass_dice_scores = validate_model_unified(
        medsam_model, valid_loader, valid_eval_loader, seg_loss, ce_loss, device
    )
    
    # 打印多类别验证结果
    if len(val_multiclass_dice_scores) > 0:
        print(f"Multi-class Validation Dice Scores:")
        for class_name, dice_score in val_multiclass_dice_scores.items():
            if not np.isnan(dice_score):
                print(f"  {class_name}: {dice_score:.4f}")
    
    epoch_end_time = time()
    epoch_time.append(epoch_end_time - epoch_start_time)
    train_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    
    train_losses.append(train_loss_reduced)
    val_losses.append(val_loss)
    val_dice_scores.append(val_dice)
    
    print(f"Epoch {epoch}: Train Loss: {train_loss_reduced:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
    
    # 记录到SwanLab
    # 记录epoch级别的训练指标
    epoch_train_log_dict = {
        "epoch": epoch,
        "Training/loss_epoch": train_loss_reduced,
        "Training/epoch_time": epoch_end_time - epoch_start_time
    }
    
    # 记录验证指标（仍以epoch为单位）
    validation_log_dict = {
        "epoch": epoch,
        "Validation/loss": val_loss,
        "Validation/dice_overall": val_dice
    }
    
    # 只在有实际最佳损失时才记录
    if best_loss is not None:
        validation_log_dict["Validation/best_loss"] = best_loss
    
    # 添加多类别Dice分数到验证指标
    if len(val_multiclass_dice_scores) > 0:
        for class_name, dice_score in val_multiclass_dice_scores.items():
            if not np.isnan(dice_score):
                validation_log_dict[f"Validation/dice_{class_name}"] = dice_score
    
    # 分别记录训练和验证指标
    swanlab.log(epoch_train_log_dict)
    swanlab.log(validation_log_dict)

    # 记录当前学习率（epoch级别）
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr:.6f}")
    swanlab.log({"epoch": epoch, "Training/learning_rate_epoch": current_lr})

    model_weights = medsam_model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss_reduced,
        "val_loss": val_loss,
        "val_dice": val_dice,
        "val_multiclass_dice": val_multiclass_dice_scores,
        "best_loss": best_loss if best_loss is not None else val_loss
    }
    
    # 保存学习率调度器状态
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    
    # 保存最佳模型
    if best_loss is None or val_loss < best_loss:
        if best_loss is None:
            print(f"Initial best validation loss: {val_loss:.4f}")
        else:
            print(f"New best validation loss: {best_loss:.4f} -> {val_loss:.4f}")
        best_loss = val_loss
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "medsam_text_prompt_best.pth"))

    # 始终保存最新的checkpoint
    torch.save(checkpoint, join(work_dir, "medsam_text_prompt_latest.pth"))
    
    # 早停检查
    if early_stopping is not None:
        if early_stopping(val_loss, medsam_model):
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            swanlab.log({"Training/early_stopped_epoch": epoch})
            break

print("Training completed!")
#endregion

#region 模块8：测试评估
# 功能：在测试集上评估最终模型性能

print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

try:
    # 加载最佳模型进行测试集评估
    print("Loading best model for test evaluation...")
    best_checkpoint = torch.load(join(work_dir, "medsam_text_prompt_best.pth"), weights_only=False)
    medsam_model.load_state_dict(best_checkpoint["model"])
    print(f"Loaded best model from epoch {best_checkpoint['epoch']} with validation loss: {best_checkpoint['best_loss']:.4f}")
    
    # 检查测试数据是否存在
    test_data_path = join(data_root, 'test')
    if not os.path.exists(test_data_path):
        print(f"Warning: Test data directory not found at {test_data_path}")
        print("Skipping test evaluation...")
        raise FileNotFoundError("Test data not available")
    
    print(f"Test data found at: {test_data_path}")
    
    # 在测试集上进行多标签评估
    print("\nEvaluating on test set...")
    # 只进行多标签评估，不计算单标签损失
    medsam_model.eval()
    all_test_dice_scores = []
    
    with torch.no_grad():
        # 对每个类别独立进行预测和评估
        test_class_predictions = {1: [], 2: [], 3: [], 4: [], 5: []}
        test_gt_data_list = []
        
        for batch in tqdm(test_eval_loader, desc="Test Multi-class Evaluation"):
            try:
                image = batch["image"].to(device)
                gt_multi = batch["gt_multi"].cpu().numpy()
                
                for batch_idx in range(image.shape[0]):
                    single_image = image[batch_idx:batch_idx+1]
                    single_gt = gt_multi[batch_idx]
                    test_gt_data_list.append(single_gt)
                        
                    # 对每个类别独立进行预测
                    for class_id in [1, 2, 3, 4, 5]:
                        class_names = test_eval_loader.dataset.lesion_names[class_id]
                        selected_name = np.random.choice(class_names)
                        
                        tokens = test_eval_loader.dataset.tokenize_text(selected_name).unsqueeze(0).to(device)
                        
                        pred = medsam_model(single_image, tokens)
                        pred_sigmoid = torch.sigmoid(pred)
                        pred_binary = (pred_sigmoid > 0.5).cpu().numpy()[0, 0]
                        
                        pred_binary_resized = cv2.resize(
                            pred_binary.astype(np.uint8),
                            (1024, 1024),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        
                        test_class_predictions[class_id].append(pred_binary_resized)
            except Exception as e:
                print(f"Error processing test batch: {str(e)}")
                continue
        
        # 计算每个类别的Dice分数
        print(f"Computing Dice scores for {len(test_gt_data_list)} test samples...")
        for i, gt_data in enumerate(test_gt_data_list):
            dice_scores_multi = {}
            label_dict = {1: "normal", 2: "GGO", 3: "reticulation", 4: "consolidation", 5: "honeycombing"}
            
            for class_id in [1, 2, 3, 4, 5]:
                try:
                    gt_class = (gt_data == class_id).astype(np.uint8)
                    pred_class = test_class_predictions[class_id][i].astype(np.uint8)
                    
                    if np.sum(gt_class) == 0 and np.sum(pred_class) == 0:
                        dice_scores_multi[label_dict[class_id]] = np.nan
                    elif np.sum(gt_class) == 0 and np.sum(pred_class) > 0:
                        dice_scores_multi[label_dict[class_id]] = 0.0
                    else:
                        dice_scores_multi[label_dict[class_id]] = compute_dice_coefficient(gt_class, pred_class)
                except Exception as e:
                    print(f"Error computing dice for class {class_id}, sample {i}: {str(e)}")
                    dice_scores_multi[label_dict[class_id]] = 0.0
            
            all_test_dice_scores.append(dice_scores_multi)
    
    # 计算测试集平均Dice分数
    test_multiclass_dice_scores = {}
    if len(all_test_dice_scores) > 0:
        for key in all_test_dice_scores[0].keys():
            scores = [score[key] for score in all_test_dice_scores if not np.isnan(score[key])]
            test_multiclass_dice_scores[key] = np.mean(scores) if len(scores) > 0 else 0.0
    
    medsam_model.train()
    
    print("\nTest Set Multi-class Dice Scores:")
    print("-" * 40)
    for class_name, dice_score in test_multiclass_dice_scores.items():
        if not np.isnan(dice_score):
            print(f"{class_name:15s}: {dice_score:.4f}")
        else:
            print(f"{class_name:15s}: N/A (no samples)")
    
    # 计算平均dice（排除NaN值）
    valid_scores = [score for score in test_multiclass_dice_scores.values() if not np.isnan(score)]
    if len(valid_scores) > 0:
        avg_test_dice = np.mean(valid_scores)
        print(f"\nAverage Test Dice (excluding N/A): {avg_test_dice:.4f}")
    else:
        print("\nNo valid dice scores computed.")
        avg_test_dice = 0.0
    
    # 保存测试结果
    test_results = {
        'test_multiclass_dice': test_multiclass_dice_scores,
        'avg_test_dice': avg_test_dice,
        'best_epoch': best_checkpoint['epoch'],
        'best_val_loss': best_checkpoint['best_loss'],
        'num_test_samples': len(test_gt_data_list)
    }
    
    print(f"\nSaving test results...")
    with open(join(work_dir, "test_results.json"), 'w') as f:
        # 将numpy类型转换为Python原生类型以便JSON序列化
        json_results = {}
        for key, value in test_results.items():
            if key == 'test_multiclass_dice':
                json_results[key] = {k: float(v) if not np.isnan(v) else None for k, v in value.items()}
            else:
                json_results[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        json.dump(json_results, f, indent=2)
    
    print(f"Test results saved to: {join(work_dir, 'test_results.json')}")
    
    # 记录测试集结果到SwanLab
    print("Logging test results to SwanLab...")
    test_log_dict = {
        "Testing/avg_dice": avg_test_dice,
        "Testing/best_epoch": best_checkpoint['epoch'],
        "Testing/final_best_val_loss": best_checkpoint['best_loss'],
        "Testing/num_test_samples": len(test_gt_data_list)
    }
    
    # 添加测试集多类别Dice分数
    for class_name, dice_score in test_multiclass_dice_scores.items():
        if not np.isnan(dice_score):
            test_log_dict[f"Testing/dice_{class_name}"] = dice_score
    
    swanlab.log(test_log_dict)
    
    print("Test evaluation completed successfully!")
    
except Exception as e:
    print(f"Error during test evaluation: {str(e)}")
    print("Test evaluation failed, but training was completed successfully")
    avg_test_dice = 0.0
    test_multiclass_dice_scores = {}
    
    # 记录失败信息到SwanLab
    try:
        swanlab.log({
            "Testing/evaluation_failed": True,
            "Testing/error_message": str(e)
        })
    except:
        pass

# 关闭SwanLab
print("\nFinalizing experiment...")
swanlab.finish()

print("\n" + "="*50)
print("TRAINING AND EVALUATION COMPLETED!")
print("="*50)
print(f"Experiment: {exp_name}")
print(f"Total epochs trained: {num_epochs}")
if 'best_checkpoint' in locals():
    print(f"Best validation loss: {best_checkpoint['best_loss']:.4f}")
    print(f"Best epoch: {best_checkpoint['epoch']}")
if 'avg_test_dice' in locals() and avg_test_dice > 0:
    print(f"Final test Dice score: {avg_test_dice:.4f}")
print(f"Model saved to: {work_dir}")
print(f"SwanLab experiment: {exp_name} finished!")
print("="*50)
#endregion
