import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_loader import create_binary_datasets
import torch

class EEGBinaryDataModule(pl.LightningDataModule):
    """EEG二分类数据模块 - 优化版"""
    
    def __init__(self, data_cfg, train_subs, val_subs, train_vids, val_vids, 
                 loo, num_workers):
        super().__init__()
        
        self.data_cfg = data_cfg
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.train_vids = train_vids
        self.val_vids = val_vids
        self.loo = loo
        self.num_workers = num_workers
        
        # 🔥 缓存数据集，避免重复加载
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        
        print(f"📂 EEG二分类数据模块初始化 (优化版):")
        print(f"   训练被试: {len(train_subs) if train_subs else 0}")
        print(f"   验证被试: {len(val_subs) if val_subs else 0}")
        print(f"   工作进程: {num_workers}")
        
    def setup(self, stage=None):
        """设置数据集 - 只加载一次"""
        if self._train_dataset is not None:
            print("📦 使用缓存的数据集")
            return
            
        print(f"🔧 数据模块设置阶段: {stage}")
        
        # 加载数据...
        self._train_dataset, self._val_dataset, self._test_dataset = create_binary_datasets(
            data_dir=self.data_cfg.data_dir,
            train_subs=self.train_subs,
            val_subs=self.val_subs,
            max_samples=getattr(self.data_cfg, 'max_samples', None),
            n_channels=self.data_cfg.n_channels,
            n_timepoints=self.data_cfg.n_timepoints,
        )
        
        # 🔥 数据质量检查
        print(f"📊 数据质量分析:")
        
        # 检查训练数据
        train_data_tensor = self._train_dataset.data
        train_labels_tensor = self._train_dataset.labels
        
        print(f"   训练数据形状: {train_data_tensor.shape}")
        # print(f"   数据类型: {train_data_tensor.dtype}")
        print(f"   数据范围: [{train_data_tensor.min():.3f}, {train_data_tensor.max():.3f}, 数据均值: {train_data_tensor.mean():.3f}]")
        # print(f"   数据均值: {train_data_tensor.mean():.3f}")
        # print(f"   数据标准差: {train_data_tensor.std():.3f}")
        
        # 🔥 检查是否所有样本都相同
        first_sample = train_data_tensor[0]
        all_same = True
        for i in range(min(10, len(train_data_tensor))):
            if not torch.allclose(train_data_tensor[i], first_sample, rtol=1e-3):
                all_same = False
                break
        
        if all_same:
            print("⚠️ 警告：前10个样本几乎相同！数据可能有问题")
        else:
            print("✅ 样本间存在差异")
        
        # 🔥 按类别检查数据差异
        pos_indices = train_labels_tensor == 1
        neg_indices = train_labels_tensor == 0
        
        if pos_indices.any() and neg_indices.any():
            pos_data = train_data_tensor[pos_indices]
            neg_data = train_data_tensor[neg_indices]
            
            pos_mean = pos_data.mean()
            neg_mean = neg_data.mean()
            
            print(f"   正样本均值: {pos_mean:.3f},负样本均值: {neg_mean:.3f}")
            print(f"   类别间差异: {abs(pos_mean - neg_mean):.3f}")
            
            if abs(pos_mean - neg_mean) < 0.01:
                print("⚠️ 警告：正负样本几乎无差异！")
            else:
                print("✅ 正负样本存在差异")
        
        print(f"✅ 数据集设置完成:")
        print(f"   训练集: {len(self._train_dataset)}")
        print(f"   验证集: {len(self._val_dataset)},测试集: {len(self._test_dataset)}")
    
    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property 
    def val_dataset(self):
        return self._val_dataset
        
    @property
    def test_dataset(self):
        return self._test_dataset
    
    def train_dataloader(self):
        # 🔥 优化DataLoader设置
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=min(self.num_workers, 4),  # 🔥 限制worker数量
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,  # 🔥 保持worker存活
            prefetch_factor=2 if self.num_workers > 0 else None,  # 🔥 预取数据
            drop_last=True  # 🔥 丢弃不完整的批次
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=min(self.num_workers, 2),  # 🔥 验证时用更少worker
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=0,  # 🔥 测试时不用多进程
            pin_memory=torch.cuda.is_available()
        )

# 保持与主项目一致的命名
EEGDataModule = EEGBinaryDataModule