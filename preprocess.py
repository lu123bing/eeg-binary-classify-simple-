import os
import scipy.io as sio
import numpy as np
import pickle
from tqdm import tqdm

def preprocess_eeg_data():
    """
    从原始.mat文件中提取每个视频的最后一秒EEG数据并打标签
    """
    
    # 🔥 配置路径
    raw_data_dir = r"E:\PyCharm Community Edition 2023.2.1\path\pypj\DAEST-main\processed_data"
    output_dir = r"E:\PyCharm Community Edition 2023.2.1\path\pypj\DAEST-main\sliced_data_1s"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 🔥 修正的视频标签映射 - 去除中性视频后的新索引
    # 注意：mat文件中的索引从0开始，所以要减1
    video_labels = {
        # 负面情绪 -> 0 (原视频1-12，现在是索引0-11)
        0: 0, 1: 0, 2: 0,      # 愤怒 (原视频1-3)
        3: 0, 4: 0, 5: 0,      # 厌恶 (原视频4-6)
        6: 0, 7: 0, 8: 0,      # 恐惧 (原视频7-9)
        9: 0, 10: 0, 11: 0,    # 悲伤 (原视频10-12)
        
        # 正面情绪 -> 1 (原视频17-28，现在是索引12-23)
        12: 1, 13: 1, 14: 1,   # 娱乐 (原视频17-19)
        15: 1, 16: 1, 17: 1,   # 激励 (原视频20-22)
        18: 1, 19: 1, 20: 1,   # 快乐 (原视频23-25)
        21: 1, 22: 1, 23: 1,   # 温柔 (原视频26-28)
    }
    
    # 🔥 原始视频ID到新索引的映射（用于文件名）
    original_video_mapping = {
        0: 1, 1: 2, 2: 3,      # 愤怒
        3: 4, 4: 5, 5: 6,      # 厌恶
        6: 7, 7: 8, 8: 9,      # 恐惧
        9: 10, 10: 11, 11: 12, # 悲伤
        12: 17, 13: 18, 14: 19, # 娱乐
        15: 20, 16: 21, 17: 22, # 激励
        18: 23, 19: 24, 20: 25, # 快乐
        21: 26, 22: 27, 23: 28, # 温柔
    }
    
    print(f"🔍 搜索原始数据目录: {raw_data_dir}")
    
    # 🔥 查找所有.mat文件
    mat_files = []
    for file in os.listdir(raw_data_dir):
        if file.endswith('.mat'):
            mat_files.append(os.path.join(raw_data_dir, file))
    
    print(f"📂 找到 {len(mat_files)} 个.mat文件")
    
    if len(mat_files) == 0:
        print("❌ 没有找到.mat文件！")
        return
    
    # 🔥 处理每个被试的数据
    metadata = {
        'subject_info': [],
        'video_info': [],
        'label_mapping': {0: 'negative', 1: 'positive'},
        'sampling_rate': 250,
        'duration': 1.0,  # 最后1秒
        'n_channels': 17,  # 根据实际情况调整
        'n_videos': 24,    # 去除中性视频后剩余24个
        'video_mapping': original_video_mapping
    }
    
    total_samples = 0
    
    for mat_file in tqdm(mat_files, desc="处理被试数据"):
        try:
            # 🔥 加载.mat文件
            mat_data = sio.loadmat(mat_file)
            
            # 提取被试ID（从文件名）
            subject_id = extract_subject_id(mat_file)
            
            print(f"\n🔄 处理被试 {subject_id}")
            print(f"📁 文件: {os.path.basename(mat_file)}")
            
            # 🔥 检查.mat文件结构
            print(f"📊 .mat文件键: {list(mat_data.keys())}")
            
            # 🔥 根据实际的mat文件结构提取数据
            eeg_data = None
            n_samples_one = None
            binary_labels = None
            
            if 'data_all_cleaned' in mat_data:
                eeg_data = mat_data['data_all_cleaned']
                print(f"✅ 找到EEG数据: data_all_cleaned, 形状={eeg_data.shape}")
            
            if 'n_samples_one' in mat_data:
                n_samples_one = mat_data['n_samples_one']
                print(f"✅ 找到样本数信息: n_samples_one, 形状={n_samples_one.shape}")
            
            if 'binary_labels' in mat_data:
                binary_labels = mat_data['binary_labels']
                print(f"✅ 找到标签信息: binary_labels, 形状={binary_labels.shape}")
            
            if eeg_data is None:
                print(f"❌ 无法找到EEG数据，跳过文件: {mat_file}")
                continue
            
            # 🔥 调整通道数（从30通道到17通道）
            if eeg_data.shape[0] == 30:
                # 选择前17个通道，或者根据实际需要选择特定通道
                eeg_data = eeg_data[:17, :]  # 取前17个通道
                print(f"📊 调整通道数: 30 -> 17, 新形状={eeg_data.shape}")
            
            # 🔥 根据数据形状和样本数信息组织数据
            video_data_list = organize_eeg_data(eeg_data, n_samples_one, expected_videos=24)
            
            if video_data_list is None:
                print(f"❌ 无法解析EEG数据结构，跳过: {mat_file}")
                continue
            
            print(f"📊 成功分割出 {len(video_data_list)} 个视频")
            
            # 🔥 处理每个视频 - 使用新的索引
            for video_idx in video_labels.keys():
                try:
                    # 提取该视频的数据
                    video_eeg = extract_video_data(video_data_list, video_idx)
                    
                    if video_eeg is None:
                        print(f"⚠️ 视频索引 {video_idx} 数据提取失败")
                        continue
                    
                    print(f"📊 视频 {video_idx} 原始形状: {video_eeg.shape}")
                    
                    # 🔥 提取最后一秒数据 (250Hz * 1秒 = 250个时间点)
                    sampling_rate = 250
                    last_second_points = sampling_rate * 1  # 最后1秒
                    
                    if video_eeg.shape[-1] < last_second_points:
                        print(f"⚠️ 视频索引 {video_idx} 数据长度不足1秒: {video_eeg.shape[-1]} < {last_second_points}")
                        continue
                    
                    # 提取最后1秒
                    last_second_eeg = video_eeg[:, -2*last_second_points:]
                    
                    # 确保形状为 (17, 250)
                    if last_second_eeg.shape[0] != 17:
                        if last_second_eeg.shape[1] == 17:
                            last_second_eeg = last_second_eeg.T
                        else:
                            # 调整通道数
                            last_second_eeg = adjust_channels(last_second_eeg, target_channels=17)
                    
                    print(f"📊 视频 {video_idx} 最后1秒形状: {last_second_eeg.shape}")
                    
                    # 🔥 获取标签和原始视频ID
                    label = video_labels[video_idx]
                    original_video_id = original_video_mapping[video_idx]
                    
                    # 🔥 保存数据 - 使用原始视频ID命名以保持一致性
                    filename = f"sub{subject_id:03d}_vid{original_video_id:02d}_idx{video_idx:02d}_label{label}.pkl"
                    filepath = os.path.join(output_dir, filename)
                    
                    # 保存为pickle格式
                    data_dict = {
                        'eeg': last_second_eeg,
                        'label': label,
                        'subject_id': subject_id,
                        'video_idx': video_idx,  # 新的索引位置
                        'original_video_id': original_video_id,  # 原始视频ID
                        'emotion': get_emotion_name(video_idx),
                        'valence': 'positive' if label == 1 else 'negative'
                    }
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(data_dict, f)
                    
                    # 🔥 更新元数据
                    metadata['video_info'].append({
                        'filename': filename,
                        'subject_id': subject_id,
                        'video_idx': video_idx,
                        'original_video_id': original_video_id,
                        'label': label,
                        'emotion': get_emotion_name(video_idx),
                        'shape': last_second_eeg.shape
                    })
                    
                    total_samples += 1
                    print(f"✅ 保存视频 {video_idx}: {filename}")
                    
                except Exception as e:
                    print(f"⚠️ 处理视频索引 {video_idx} 时出错: {e}")
                    continue
            
            # 🔥 记录被试信息
            processed_videos = len([info for info in metadata['video_info'] if info['subject_id'] == subject_id])
            metadata['subject_info'].append({
                'subject_id': subject_id,
                'filename': os.path.basename(mat_file),
                'processed_videos': processed_videos
            })
            
            print(f"📊 被试 {subject_id} 处理完成: {processed_videos} 个视频")
            
        except Exception as e:
            print(f"❌ 处理文件 {mat_file} 时出错: {e}")
            continue
    
    # 🔥 保存元数据
    metadata_file = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n🎉 数据预处理完成!")
    print(f"📊 处理结果:")
    print(f"   总样本数: {total_samples}")
    print(f"   被试数: {len(metadata['subject_info'])}")
    print(f"   输出目录: {output_dir}")
    print(f"   元数据文件: {metadata_file}")
    
    # 🔥 统计标签分布
    labels = [info['label'] for info in metadata['video_info']]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   标签分布: {dict(zip(unique, counts))}")

def extract_subject_id(mat_file):
    """从文件名提取被试ID"""
    filename = os.path.basename(mat_file)
    
    # 常见的被试ID模式
    import re
    
    # 模式1: sub001.mat, subject_001.mat
    match = re.search(r'sub(?:ject)?[_-]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 模式2: 001.mat, 1.mat
    match = re.search(r'^(\d+)\.mat', filename)
    if match:
        return int(match.group(1))
    
    # 模式3: 任何数字
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # 默认使用文件名哈希
    return hash(filename) % 1000

def organize_eeg_data(eeg_data, n_samples_one=None, expected_videos=24):
    """组织EEG数据格式 - 处理2D连续数据"""
    
    print(f"📊 组织EEG数据，原始形状: {eeg_data.shape}")
    
    if eeg_data.ndim == 2:
        # 2D数据: (channels, timepoints)
        n_channels, total_timepoints = eeg_data.shape
        print(f"📊 检测到2D数据: {n_channels}通道, {total_timepoints}时间点")
        
        # 🔥 使用n_samples_one信息来分割视频
        if n_samples_one is not None:
            print(f"📊 每个视频的样本数: {n_samples_one}")
            
            if n_samples_one.ndim == 2:
                n_samples_one = n_samples_one.flatten()
            
            if len(n_samples_one) != expected_videos:
                print(f"⚠️ 视频数量不匹配: {len(n_samples_one)} != {expected_videos}")
                return None
            
            # 🔥 根据每个视频的样本数分割数据
            organized_data = []
            current_idx = 0
            
            for i, n_samples in enumerate(n_samples_one):
                n_samples = int(n_samples)
                end_idx = current_idx + n_samples
                
                if end_idx > total_timepoints:
                    print(f"⚠️ 视频 {i} 数据超出范围: {end_idx} > {total_timepoints}")
                    break
                
                # 提取该视频的数据
                video_data = eeg_data[:, current_idx:end_idx]
                organized_data.append(video_data)
                
                print(f"📊 视频 {i}: 样本数={n_samples}, 形状={video_data.shape}")
                current_idx = end_idx
            
            return organized_data
        
        else:
            # 🔥 平均分割（假设每个视频30秒，250Hz）
            video_length = 250 * 30  # 30秒
            
            if total_timepoints >= expected_videos * video_length:
                organized_data = []
                for i in range(expected_videos):
                    start_idx = i * video_length
                    end_idx = start_idx + video_length
                    video_data = eeg_data[:, start_idx:end_idx]
                    organized_data.append(video_data)
                return organized_data
            else:
                print(f"⚠️ 数据长度不足{expected_videos}个视频: {total_timepoints} < {expected_videos * video_length}")
                return None
    
    elif eeg_data.ndim == 3:
        # 3D数据处理保持不变
        print(f"📊 检测到3D数据: {eeg_data.shape}")
        if eeg_data.shape[0] == expected_videos:
            return [eeg_data[i] for i in range(expected_videos)]
        elif eeg_data.shape[1] == expected_videos:
            return [eeg_data[:, i, :] for i in range(expected_videos)]
        elif eeg_data.shape[2] == expected_videos:
            return [eeg_data[:, :, i] for i in range(expected_videos)]
    
    print(f"❌ 无法识别的EEG数据格式: {eeg_data.shape}")
    return None

def extract_video_data(eeg_data_list, video_idx):
    """从数据列表中提取特定视频的EEG数据"""
    
    if isinstance(eeg_data_list, list):
        if 0 <= video_idx < len(eeg_data_list):
            return eeg_data_list[video_idx]
        else:
            print(f"⚠️ 视频索引超出范围: {video_idx} >= {len(eeg_data_list)}")
            return None
    
    return None

def adjust_channels(eeg_data, target_channels=17):
    """调整EEG数据的通道数"""
    
    current_channels = eeg_data.shape[0]
    
    if current_channels == target_channels:
        return eeg_data
    elif current_channels > target_channels:
        # 截取前target_channels个通道
        return eeg_data[:target_channels, :]
    else:
        # 用零填充或重复通道
        padded_data = np.zeros((target_channels, eeg_data.shape[1]))
        padded_data[:current_channels, :] = eeg_data
        
        # 重复最后几个通道填充
        if current_channels > 0:
            for i in range(current_channels, target_channels):
                padded_data[i, :] = eeg_data[i % current_channels, :]
        
        return padded_data

def get_emotion_name(video_idx):
    """根据新的视频索引获取情绪名称"""
    emotion_mapping = {
        # 负面情绪
        0: 'anger', 1: 'anger', 2: 'anger',
        3: 'disgust', 4: 'disgust', 5: 'disgust',
        6: 'fear', 7: 'fear', 8: 'fear',
        9: 'sadness', 10: 'sadness', 11: 'sadness',
        
        # 正面情绪  
        12: 'amusement', 13: 'amusement', 14: 'amusement',
        15: 'inspiration', 16: 'inspiration', 17: 'inspiration',
        18: 'joy', 19: 'joy', 20: 'joy',
        21: 'tenderness', 22: 'tenderness', 23: 'tenderness',
    }
    return emotion_mapping.get(video_idx, 'unknown')

def verify_processed_data(output_dir):
    """验证处理后的数据"""
    print(f"\n🔍 验证处理后的数据...")
    
    # 加载元数据
    metadata_file = os.path.join(output_dir, 'metadata.pkl')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"📊 元数据信息:")
        print(f"   被试数: {len(metadata['subject_info'])}")
        print(f"   视频样本数: {len(metadata['video_info'])}")
        
        # 验证几个样本文件
        sample_files = metadata['video_info'][:5]
        for info in sample_files:
            filepath = os.path.join(output_dir, info['filename'])
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ {info['filename']}: EEG形状={data['eeg'].shape}, 标签={data['label']}")
            else:
                print(f"❌ 文件不存在: {info['filename']}")

if __name__ == "__main__":
    preprocess_eeg_data()
    
    # 验证处理结果
    output_dir = r"E:\PyCharm Community Edition 2023.2.1\path\pypj\DAEST-main\sliced_data"
    verify_processed_data(output_dir)