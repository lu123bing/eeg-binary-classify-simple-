# cnn_svm/data_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

def create_binary_datasets(data_dir=None, train_subs=None, val_subs=None, max_samples=None, 
                          n_channels=17, n_timepoints=250):
    """创建二分类数据集 - 简单参数版本"""
    
    print("🔄 创建二分类数据集 (使用预处理数据)...")
    print(f"   数据目录: {data_dir}")
    print(f"   配置: {n_channels}通道, {n_timepoints}时间点")
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return create_dummy_datasets(n_channels, n_timepoints)
    
    # 加载数据
    train_dataset, val_dataset, test_dataset = load_preprocessed_eeg_data(
        data_dir, train_subs, val_subs, max_samples, n_channels, n_timepoints
    )
    
    return train_dataset, val_dataset, test_dataset

def load_preprocessed_eeg_data(data_dir, train_subs=None, val_subs=None, max_samples=None,
                              n_channels=17, n_timepoints=500):
    """加载预处理后的EEG数据"""
    
    # 加载元数据
    metadata_file = os.path.join(data_dir, 'metadata.pkl')
    if not os.path.exists(metadata_file):
        print(f"❌ 元数据文件不存在: {metadata_file}")
        return create_dummy_datasets(n_channels, n_timepoints)
    
    print(f"📂 加载元数据: {metadata_file}")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    # print(f"📊 元数据信息:")
    # print(f"   被试数: {len(metadata['subject_info'])}")
    # print(f"   视频样本数: {len(metadata['video_info'])}")
    # print(f"   标签映射: {metadata['label_mapping']}")
    
    # 根据被试划分过滤数据
    if train_subs is not None or val_subs is not None:
        train_files, val_files = filter_by_subjects(metadata['video_info'], train_subs, val_subs)
    else:
        # 没有指定被试划分，使用所有数据
        all_files = metadata['video_info']
        train_files = val_files = all_files
    
    # print(f"📊 数据文件统计:")
    # print(f"   训练文件: {len(train_files) if train_files else 0}")
    # print(f"   验证文件: {len(val_files) if val_files else 0}")
    
    # 加载数据文件
    if train_files:
        train_data, train_labels = load_data_files(
            data_dir, train_files, max_samples, "训练", n_channels, n_timepoints
        )
    else:
        train_data, train_labels = [], []
    
    # 🔥 加载验证数据
    if val_files and train_subs != val_subs:  # 避免重复加载相同数据
        val_data, val_labels = load_data_files(
            data_dir, val_files, max_samples, "验证", n_channels, n_timepoints
        )
    else:
        val_data, val_labels = train_data, train_labels
    
    # 🔥 检查数据是否为空
    if len(train_data) == 0:
        print("⚠️ 没有训练数据，使用模拟数据")
        return create_dummy_datasets(n_channels, n_timepoints)
    
    # 转换为numpy数组
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    val_data = np.array(val_data, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.int64)
    
    print(f"📊 加载后数据统计:")
    print(f"   训练数据形状: {train_data.shape},标签分布: {np.bincount(train_labels)}")
    print(f"   训练标签分布: {np.bincount(train_labels)}")
    # print(f"   验证数据形状: {val_data.shape}")
    # print(f"   验证标签分布: {np.bincount(val_labels)}")
    
    # 如果验证数据和训练数据相同，进行划分
    if np.array_equal(train_data, val_data):
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, train_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=train_labels
        )
        print(f"📊 重新划分后:")
        print(f"   训练数据: {train_data.shape}")
        print(f"   验证数据: {val_data.shape}")
    
    # 创建测试集（从验证集分离）
    if len(val_data) > 100:  # 确保有足够数据分离测试集
        val_data, test_data, val_labels, test_labels = train_test_split(
            val_data, val_labels,
            test_size=0.5,
            random_state=42,
            stratify=val_labels
        )
    else:
        test_data, test_labels = val_data.copy(), val_labels.copy()
    
    # print(f"📊 最终数据集划分:")
    # print(f"   训练集: {len(train_data)} 样本")
    # print(f"   验证集: {len(val_data)} 样本") 
    # print(f"   测试集: {len(test_data)} 样本")
    
    # 创建数据集
    train_dataset = EEGBinaryDataset(train_data, train_labels, "训练")
    val_dataset = EEGBinaryDataset(val_data, val_labels, "验证")
    test_dataset = EEGBinaryDataset(test_data, test_labels, "测试")
    
    return train_dataset, val_dataset, test_dataset

def filter_by_subjects(video_info_list, train_subs, val_subs):
    """根据被试ID过滤数据文件"""
    
    train_files = []
    val_files = []
    
    # 🔥 确保被试列表是列表格式
    if train_subs is not None:
        if isinstance(train_subs, np.ndarray):
            train_subs = train_subs.tolist()
        elif not isinstance(train_subs, list):
            train_subs = [train_subs]
    
    if val_subs is not None:
        if isinstance(val_subs, np.ndarray):
            val_subs = val_subs.tolist()
        elif not isinstance(val_subs, list):
            val_subs = [val_subs]
    
    # 🔥 遍历所有视频信息，根据被试ID分类
    for info in video_info_list:
        subject_id = info['subject_id']
        
        if train_subs is not None and subject_id in train_subs:
            train_files.append(info)
        elif val_subs is not None and subject_id in val_subs:
            val_files.append(info)
        elif train_subs is None and val_subs is None:
            # 如果没有指定划分，所有数据都用于训练和验证
            train_files.append(info)
            val_files.append(info)
    
    print(f"📊 被试筛选结果:")
    if train_subs:
        train_subjects = set(info['subject_id'] for info in train_files)
        # print(f"   训练被试: {sorted(train_subjects)} ({len(train_subjects)}个)")
    if val_subs:
        val_subjects = set(info['subject_id'] for info in val_files)
        print(f"   验证被试: {sorted(val_subjects)} ({len(val_subjects)}个)")
    
    return train_files, val_files

def load_data_files(data_dir, file_info_list, max_samples=None, dataset_name="",
                   n_channels=17, n_timepoints=500):
    """加载指定的数据文件"""
    
    if max_samples and len(file_info_list) > max_samples:
        # 随机选择文件
        np.random.seed(42)
        selected_indices = np.random.choice(len(file_info_list), max_samples, replace=False)
        file_info_list = [file_info_list[i] for i in selected_indices]
        print(f"⚠️ {dataset_name}数据限制加载: {max_samples} 个文件")
    
    all_data = []
    all_labels = []
    success_count = 0
    failed_count = 0
    
    print(f"📥 加载{dataset_name}数据: {len(file_info_list)} 个文件")
    
    expected_shape = (n_channels, n_timepoints)
    
    for info in tqdm(file_info_list, desc=f"加载{dataset_name}数据"):
        try:
            file_path = os.path.join(data_dir, info['filename'])
            
            if not os.path.exists(file_path):
                failed_count += 1
                continue
            
            # 🔥 加载预处理后的pkl文件
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            # 🔥 提取EEG数据和标签
            if 'eeg' in data_dict and 'label' in data_dict:
                eeg_signal = data_dict['eeg']
                label = data_dict['label']
                
                # 确保数据格式正确
                if isinstance(eeg_signal, np.ndarray) and eeg_signal.shape == expected_shape:
                    all_data.append(eeg_signal.astype(np.float32))
                    all_labels.append(int(label))
                    success_count += 1
                else:
                    print(f"⚠️ 数据形状不正确: {info['filename']}, 形状: {eeg_signal.shape}, 期望: {expected_shape}")
                    failed_count += 1
            else:
                print(f"⚠️ 数据格式不正确: {info['filename']}")
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # 只打印前5个错误
                print(f"⚠️ 加载失败 {info['filename']}: {e}")
    
    print(f"📊 {dataset_name}数据加载结果:")
    print(f"   成功: {success_count} 个文件")
    print(f"   失败: {failed_count} 个文件")
    
    return all_data, all_labels

def create_dummy_datasets(n_channels=17, n_timepoints=500):
    """创建模拟数据集用于测试"""
    print("🎯 创建模拟EEG数据集...")
    print(f"   参数: {n_channels}通道, {n_timepoints}时间点")
    
    # 模拟数据
    n_samples = 5000
    
    # 生成随机EEG数据
    np.random.seed(42)
    dummy_data = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    
    # 🔥 添加一些真实的EEG特征模式
    for i in range(n_samples):
        # 添加α波(8-12Hz)和β波(13-30Hz)模式
        t = np.linspace(0, n_timepoints/250, n_timepoints)  # 假设250Hz采样率
        alpha_wave = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10Hz α波
        beta_wave = 0.3 * np.sin(2 * np.pi * 20 * t)   # 20Hz β波
        
        # 随机选择几个通道添加波形
        channels = np.random.choice(n_channels, min(5, n_channels), replace=False)
        for ch in channels:
            dummy_data[i, ch] += alpha_wave + beta_wave
    
    # 生成二分类标签 (0: 消极, 1: 积极)
    dummy_labels = np.random.randint(0, 2, n_samples).astype(np.int64)
    
    print(f"📊 模拟数据信息:")
    print(f"   数据形状: {dummy_data.shape}")
    print(f"   标签分布: {np.bincount(dummy_labels)}")
    
    # 划分数据集
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        dummy_data, dummy_labels, test_size=0.2, random_state=42, stratify=dummy_labels
    )
    
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # 创建数据集
    train_dataset = EEGBinaryDataset(train_data, train_labels, "训练(模拟)")
    val_dataset = EEGBinaryDataset(val_data, val_labels, "验证(模拟)")
    test_dataset = EEGBinaryDataset(test_data, test_labels, "测试(模拟)")
    
    return train_dataset, val_dataset, test_dataset

class EEGBinaryDataset(Dataset):
    """EEG二分类数据集"""
    
    def __init__(self, data, labels, name=""):
        self.data = torch.FloatTensor(data)  # (N, 17, 500)
        self.labels = torch.LongTensor(labels)  # (N,)
        self.name = name
        
        print(f"📦 {name}数据集初始化:")
        print(f"   数据: {self.data.shape}, dtype: {self.data.dtype}")
        print(f"   标签: {self.labels.shape}, dtype: {self.labels.dtype}")
        print(f"   标签分布: {np.bincount(self.labels.numpy())}")
        print(f"   数据范围: [{self.data.min():.3f}, {self.data.max():.3f}]")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]