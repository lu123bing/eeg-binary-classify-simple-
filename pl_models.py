import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class BinaryExtractorModel(pl.LightningModule):
    """二分类特征提取器模型 - 修复为BCEWithLogitsLoss"""
    
    def __init__(self, model, cfg_train):
        super().__init__()
        self.model = model
        self.cfg_train = cfg_train
        
        # 🔥 使用BCEWithLogitsLoss，对混合精度安全
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 🔥 用于收集验证结果
        self.validation_step_outputs = []
        
        # 🔥 手动跟踪最佳指标
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_metrics = {}
        
    def forward(self, x):
        return self.model.predict(x)  # 返回logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # 🔥 增强数据增强
        if self.training:
            x = self.add_noise(x, noise_factor=0.1)  # 🔥 增加噪声
    
        logits = self.forward(x).squeeze(1)
        y_float = y.float()
        
        # 🔥 添加L2正则化到损失
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param, p=2)
    
        base_loss = self.criterion(logits, y_float)
        loss = base_loss + 0.01 * l2_reg  # 🔥 添加L2惩罚
    
        # 计算准确率
        probs = torch.sigmoid(logits)
        pred_labels = (probs > 0.5).float()
        acc = (pred_labels == y_float).float().mean()
    
        self.log_dict({
            'ext/train/loss': loss,
            'ext/train/base_loss': base_loss,
            # 'ext/train/l2_reg': l2_reg,
            'ext/train/acc': acc,
        }, on_step=False, on_epoch=True, prog_bar=True)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # 🔥 获取logits
        logits = self.forward(x)
        logits = logits.squeeze(1)
        
        # 🔥 标签转换
        y_float = y.float()
        
        # 🔥 计算损失
        loss = self.criterion(logits, y_float)
        
        # 🔥 计算准确率
        probs = torch.sigmoid(logits)
        pred_labels = (probs > 0.5).float()
        acc = (pred_labels == y_float).float().mean()
        
        # 🔥 记录结果
        self.validation_step_outputs.append({
            'val_loss': loss,
            'val_acc': acc,
            'predictions': pred_labels,
            'targets': y_float,
            'probabilities': probs,
            'logits': logits
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return
        
        # 🔥 收集所有预测和目标
        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        all_probs = torch.cat([x['probabilities'] for x in self.validation_step_outputs])
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        
        # 🔥 寻找最优阈值
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        targets_np = all_targets.cpu().numpy()
        probs_np = all_probs.cpu().numpy()
        
        # 方法1：使用F1分数寻找最优阈值
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.55
        best_f1 = 0
        
        for threshold in thresholds:
            pred_at_thresh = (probs_np > threshold).astype(float)
            if len(np.unique(pred_at_thresh)) > 1:  # 确保有两个类别
                f1 = f1_score(targets_np, pred_at_thresh, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        # 🔥 使用最优阈值重新计算指标
        optimal_preds = (probs_np > best_threshold).astype(float)
        
        precision = precision_score(targets_np, optimal_preds, average='binary', zero_division=0)
        recall = recall_score(targets_np, optimal_preds, average='binary', zero_division=0)
        f1 = f1_score(targets_np, optimal_preds, average='binary', zero_division=0)
        acc = accuracy_score(targets_np, optimal_preds)
        
        # 🔥 计算AUC
        try:
            auc = roc_auc_score(targets_np, probs_np) if len(np.unique(targets_np)) > 1 else 0.0
        except:
            auc = 0.0
        
        # 🔥 计算平均损失
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        
        # 🔥 记录指标（使用最优阈值）
        self.log_dict({
            'ext/val/loss': avg_loss,
            'ext/val/acc': acc,
            'ext/val/acc_optimal': acc,
            # 'ext/val/precision': precision,
            # 'ext/val/recall': recall,
            # 'ext/val/f1': f1,
            # 'ext/val/auc': auc,
            'ext/val/optimal_threshold': best_threshold,
        }, on_epoch=True, prog_bar=True)
        
        # 🔥 打印详细结果
        print(f"\n📊 验证结果 (阈值优化):")
        print(f"   最优阈值: {best_threshold:.3f}，准确率: {acc:.4f}")
        # print(f"   精确率: {precision:.4f}")
        print(f"   召回率: {recall:.4f}，平均概率: {all_probs.mean():.4f}")
        # print(f"   F1分数: {f1:.4f}")
        # print(f"   AUC: {auc:.4f}")
        # print(f"   平均logits: {all_logits.mean():.4f}")
        print(f"   正样本比例: {targets_np.mean():.4f}")
        print(f"   预测正样本比例: {optimal_preds.mean():.4f}")
        
        # 🔥 概率分布分析
        pos_probs = probs_np[targets_np == 1]
        neg_probs = probs_np[targets_np == 0]
        print(f"   正样本平均概率: {pos_probs.mean():.4f} ± {pos_probs.std():.4f}")
        print(f"   负样本平均概率: {neg_probs.mean():.4f} ± {neg_probs.std():.4f}")
        
        # 🔥 手动记录最佳指标
        if acc > self.best_val_acc:
            self.best_val_acc = float(acc)
            self.best_val_loss = float(avg_loss)
            self.best_epoch = self.current_epoch
            self.best_metrics = {
                'val_acc': float(acc),
                'val_loss': float(avg_loss),
                'val_recall': float(recall),
                'optimal_threshold': float(best_threshold),
                'epoch': self.current_epoch
            }
            
            print(f"🎯 新的最佳验证结果! Epoch {self.current_epoch + 1}")
            print(f"   验证准确率: {self.best_val_acc:.4f}")
            print(f"   验证损失: {self.best_val_loss:.4f}")
            print(f"   最优阈值: {self.best_metrics['optimal_threshold']:.4f}")
        
        # 清空结果
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # 🔥 大幅降低学习率，增加正则化
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg_train.lr * 0.1,  # 🔥 降低学习率到1/10
            weight_decay=self.cfg_train.weight_decay * 5,  # 🔥 增加权重衰减
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 🔥 使用更保守的调度器
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=3,  # 每3个epoch降低学习率
            gamma=0.7     # 乘以0.7
        )
        
        return [optimizer], [scheduler]
    
    def add_noise(self, x, noise_factor=0.1):
        """增强数据增强"""
        if self.training:
            # 高斯噪声
            noise = torch.randn_like(x) * noise_factor
            x = x + noise
            
            # 🔥 随机mask一些通道
            if torch.rand(1).item() < 0.3:  # 30%概率
                n_channels = x.size(1)
                mask_channels = torch.randperm(n_channels)[:n_channels//4]  # mask 1/4通道
                x[:, mask_channels, :] *= 0.5
            
            return x
        return x

# 保持与主项目一致的命名
ExtractorModel = BinaryExtractorModel