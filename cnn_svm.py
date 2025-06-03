# cnn_svm/eeg_binary_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import logging
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from data_loader import create_data_loaders

# 🔥 简单检查
print(f"🔍 CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

log = logging.getLogger(__name__)

class EEGBinaryDataset(Dataset):
    """简单的EEG数据集"""
    
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]  # (17, 250)
        y = self.labels[idx]
        return x, y

class SimpleEEGNet(nn.Module):
    """简洁的EEGNet - 专门为EEG设计"""
    
    def __init__(self, n_channels=17, n_timepoints=250, n_classes=2, dropout=0.25):
        super().__init__()
        
        # 🔥 EEGNet Block 1 - 时间卷积
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
        )
        
        # 🔥 深度卷积 - 每个通道单独卷积
        self.depthwise = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        # 🔥 EEGNet Block 2 - 可分离卷积
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        # 🔥 计算特征维度
        self.feature_dim = self._get_feature_dim(n_channels, n_timepoints)
        
        # 🔥 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, n_classes)
        )
        
    def _get_feature_dim(self, n_channels, n_timepoints):
        """计算输出特征维度"""
        x = torch.randn(1, 1, n_channels, n_timepoints)
        with torch.no_grad():
            x = self.block1(x)
            x = self.depthwise(x)
            x = self.block2(x)
        return x.numel()
    
    def forward(self, x):
        # 输入: (batch, 17, 250)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, 17, 250)
        
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.block2(x)
        x = self.classifier(x)
        
        return x

class EEGBinaryModel(pl.LightningModule):
    """简单的EEG二分类模型"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # 🔥 使用简单的EEGNet
        self.model = SimpleEEGNet(
            n_channels=cfg.model.n_channels,
            n_timepoints=cfg.model.n_timepoints,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"📊 模型参数量: {sum(p.numel() for p in self.parameters()):,}")
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log_dict({
            'train/loss': loss,
            'train/acc': acc,
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log_dict({
            'val/loss': loss,
            'val/acc': acc,
        }, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {'preds': preds, 'targets': y}
    
    def configure_optimizers(self):
        # 🔥 简单的Adam优化器
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay
        )
        
        # 🔥 简单的学习率调度
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
        
        return [optimizer], [scheduler]

@hydra.main(config_path="../cfgs", config_name="binary_config", version_base="1.3")
def train_binary_classifier(cfg: DictConfig):
    """训练简单的EEG二分类器"""
    
    print("🚀 开始训练简单EEG分类器")
    
    # 设置随机种子
    pl.seed_everything(cfg.seed)
    
    # 🔥 简单的数据加载
    data_dir = cfg.data.data_dir
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        max_samples=cfg.data.get('max_samples', None)
    )
    
    print(f"📊 数据统计:")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    
    # 创建模型
    model = EEGBinaryModel(cfg)
    
    # 🔥 简单的回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:02d}-{val_acc:.3f}'
    )
    
    early_stopping = EarlyStopping(
        monitor='val/acc',
        mode='max',
        patience=cfg.train.patience,
        verbose=True
    )
    
    # 🔥 简单的日志
    wandb_logger = WandbLogger(
        name=f"simple_eegnet_{cfg.model.n_channels}ch",
        project="simple_eeg_classification"
    )
    
    # 🔥 简单的训练器
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=cfg.train.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )
    
    print("🎯 开始训练...")
    trainer.fit(model, train_loader, val_loader)
    
    print("🧪 开始测试...")
    test_results = trainer.test(model, test_loader, ckpt_path='best')
    
    # 🔥 简单的测试评估
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            if torch.cuda.is_available():
                x = x.cuda()
                
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
    
    # 计算最终准确率
    final_acc = accuracy_score(all_targets, all_preds)
    print(f"\n🎉 最终测试准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    
    # 🔥 简单的分类报告
    print("\n📊 分类报告:")
    print(classification_report(
        all_targets, all_preds, 
        target_names=['负面', '正面'],
        digits=4
    ))
    
    wandb.finish()
    return model, final_acc

if __name__ == "__main__":
    train_binary_classifier()