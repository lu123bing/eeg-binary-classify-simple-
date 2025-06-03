import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGBinaryModel(nn.Module):
    """EEG二分类模型 - 修复为logits输出"""
    
    def __init__(self, n_channels=17, n_timepoints=250, d_model=128, n_classes=2, dropout=0.1):
        super().__init__()
        
        # CNN编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            
            nn.Flatten(),
            nn.Linear(256 * 16, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 🔥 修复：输出logits（不用sigmoid）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1)  # 🔥 输出原始logits，不用sigmoid
        )
        
        self.stratified = []
        
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        features = self.encoder(x)
        return features
    
    def predict(self, x):
        """预测阶段：返回logits"""
        features = self.forward(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        """返回概率"""
        logits = self.predict(x)
        probs = torch.sigmoid(logits)
        return probs
    
    def classify(self, x):
        """分类阶段：返回类别标签"""
        probs = self.predict_proba(x)
        return (probs > 0.5).float() 

class EEGNet(nn.Module):
    """EEGNet架构 - 修复为logits输出"""
    def __init__(self, n_channels=17, n_timepoints=250, n_classes=2, dropout=0.25):
        super().__init__()
        
        # EEGNet结构保持不变...
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        self.feature_dim = self._get_feature_dim(n_channels, n_timepoints)
        
        # 🔥 修复：输出logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # 不用sigmoid
        )
        
        # 🔥 重要：初始化最后一层bias为负值
        # 这样初始预测会偏向负类，避免总是预测正类
        with torch.no_grad():
            self.classifier[-1].bias.fill_(-0.5)  # 🔥 负偏置
            self.classifier[-1].weight.normal_(0, 0.01)  # 🔥 小权重
        
        self.stratified = []
        
    def _get_feature_dim(self, n_channels, n_timepoints):
        x = torch.randn(1, 1, n_channels, n_timepoints)
        x = self.block1(x)
        x = self.block2(x)
        return x.numel()
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.block1(x)
        x = self.block2(x)
        features = x.view(x.size(0), -1)
        return features
    
    def predict(self, x):
        features = self.forward(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        logits = self.predict(x)
        probs = torch.sigmoid(logits)
        return probs
    
    def classify(self, x):
        probs = self.predict_proba(x)
        return (probs > 0.5).float()

# 🔥 为所有其他模型也添加相同的修复
class AttentionEEGNet(nn.Module):
    def __init__(self, n_channels=17, n_timepoints=250, n_classes=2, dropout=0.25):
        super().__init__()
        
        # 网络结构保持不变...
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(n_channels, n_channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(n_channels // 4, n_channels, 1),
            nn.Sigmoid()
        )
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, 25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, 15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, 10, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
        # 🔥 修复分类器 - 确保最后一层是Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # 🔥 输出logits，不要sigmoid
        )
        
        # 🔥 安全的权重初始化
        self._initialize_weights()
        
        self.stratified = []
    
    def _initialize_weights(self):
        """安全的权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 🔥 最后一层特殊初始化
        final_layer = None
        for layer in reversed(list(self.classifier.modules())):
            if isinstance(layer, nn.Linear):
                final_layer = layer
                break
        
        if final_layer is not None:
            with torch.no_grad():
                final_layer.bias.fill_(-0.5)
                final_layer.weight.normal_(0, 0.01)
    
    def forward(self, x):
        """前向传播 - 返回特征"""
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # 空间注意力
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        # 时间卷积
        x = self.temporal_conv(x)  # (B, 256, 16)
        
        # 时间注意力
        # 全局平均池化获取时间特征
        temporal_features = x.mean(dim=2)  # (B, 256)
        temporal_att = self.temporal_attention(temporal_features)  # (B, 256)
        temporal_att = temporal_att.unsqueeze(2)  # (B, 256, 1)
        
        # 应用时间注意力
        x = x * temporal_att  # (B, 256, 16)
        
        # 返回展平的特征
        features = x.view(x.size(0), -1)  # (B, 256*16)
        return features
    
    # 🔥 添加缺失的方法
    def predict(self, x):
        """预测阶段：返回logits"""
        features = self.forward(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        """返回概率"""
        logits = self.predict(x)
        probs = torch.sigmoid(logits)
        return probs
    
    def classify(self, x):
        """分类阶段：返回类别标签"""
        probs = self.predict_proba(x)
        return (probs > 0.5).float()

class ImprovedEEGNet(nn.Module):
    def __init__(self, n_channels=17, n_timepoints=250, n_classes=2, dropout=0.25):
        super().__init__()
        
        # 🔥 时间维度卷积 - 捕获时间特征
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=10, padding=5),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # 🔥 计算特征维度
        self.feature_dim = 128 * (n_timepoints // 8)
        
        # 🔥 修复分类器 - 移除Sigmoid，只保留Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # 🔥 最后一层是Linear，输出logits
        )
        
        # 🔥 正确的初始化方式
        self._initialize_weights()
        
        self.stratified = []
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 🔥 最后一层特殊初始化（负偏置）
        if isinstance(self.classifier[-1], nn.Linear):
            with torch.no_grad():
                self.classifier[-1].bias.fill_(-0.5)  # 🔥 现在可以安全访问bias
                self.classifier[-1].weight.normal_(0, 0.01)
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # 时间维度卷积
        x = self.temporal_conv(x)
        
        # 返回特征
        features = x.view(x.size(0), -1)
        return features
    
    def predict(self, x):
        features = self.forward(x)
        logits = self.classifier(features)  # 🔥 返回logits
        return logits
    
    def predict_proba(self, x):
        logits = self.predict(x)
        probs = torch.sigmoid(logits)  # 🔥 在这里应用sigmoid
        return probs
    
    def classify(self, x):
        probs = self.predict_proba(x)
        return (probs > 0.5).float()

class AdvancedEEGNet(nn.Module):
    """更高级的EEG网络 - 集成多种先进技术"""
    def __init__(self, n_channels=17, n_timepoints=250, n_classes=2, dropout=0.2):
        super().__init__()
        
        # 🔥 多尺度时间卷积
        self.temporal_conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.temporal_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.temporal_conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # 🔥 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        
        # 🔥 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # 🔥 输出logits
        )
        
        # 🔥 添加权重初始化
        self._initialize_weights()
        
        self.stratified = []
    
    def _initialize_weights(self):
        """权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 🔥 最后一层特殊初始化
        if isinstance(self.classifier[-1], nn.Linear):
            with torch.no_grad():
                self.classifier[-1].bias.fill_(-0.5)
                self.classifier[-1].weight.normal_(0, 0.01)
        
        print("✅ AdvancedEEGNet 权重初始化完成")
    
    def forward(self, x):
        """前向传播"""
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # 多尺度时间卷积
        x = self.temporal_conv1(x)  # (B, 64, T/2)
        x = self.temporal_conv2(x)  # (B, 128, T/4)
        x = self.temporal_conv3(x)  # (B, 256, T/8)
        
        # 自适应池化
        x = self.adaptive_pool(x)   # (B, 256, 32)
        
        # 展平
        features = x.view(x.size(0), -1)  # (B, 256*32)
        return features
    
    def predict(self, x):
        """预测logits"""
        features = self.forward(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        """预测概率"""
        logits = self.predict(x)
        probs = torch.sigmoid(logits)
        return probs
    
    def classify(self, x):
        """分类预测"""
        probs = self.predict_proba(x)
        return (probs > 0.5).float()

class ResidualEEGNet(nn.Module):
    """带残差连接的EEG网络"""
    def __init__(self, n_channels=17, n_timepoints=250, n_classes=2, dropout=0.1):
        super().__init__()
        
        # 🔥 输入投影
        self.input_proj = nn.Conv1d(n_channels, 64, 1)
        
        # 🔥 残差块
        self.res_blocks = nn.ModuleList([
            self._make_res_block(64, 64, 7),
            self._make_res_block(64, 128, 5),
            self._make_res_block(128, 256, 3),
        ])
        
        # 🔥 注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
        # 🔥 修复分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(256 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # 🔥 输出logits
        )
        
        self._initialize_weights()
        self.stratified = []
    
    def _make_res_block(self, in_channels, out_channels, kernel_size):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
        )
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # 输入投影
        x = self.input_proj(x)
        
        # 残差块
        for i, block in enumerate(self.res_blocks):
            identity = x
            out = block(x)
            
            # 调整维度用于残差连接
            if identity.shape[1] != out.shape[1]:
                identity = F.interpolate(identity, size=out.shape[2])
                identity = nn.Conv1d(identity.shape[1], out.shape[1], 1).to(x.device)(identity)
            
            x = F.relu(out + identity)
            x = F.max_pool1d(x, 2)
        
        # 注意力
        att_weights = self.attention(x).unsqueeze(-1)
        x = x * att_weights
        
        return x.view(x.size(0), -1)
    
    def predict(self, x):
        features = self.forward(x)
        logits = self.classifier(features)
        return logits

def create_binary_model(cfg):
    """创建优化的模型"""
    model_type = getattr(cfg.model, 'model_type', 'improved_eegnet')
    
    if model_type == 'improved_eegnet':
        model = ImprovedEEGNet(
            n_channels=cfg.model.n_channels,
            n_timepoints=cfg.model.n_timepoints,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout
        )
    elif model_type == 'advanced':
        model = AdvancedEEGNet(
            n_channels=cfg.model.n_channels,
            n_timepoints=cfg.model.n_timepoints,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout
        )
    elif model_type == 'residual_eegnet':
        model = ResidualEEGNet(
            n_channels=cfg.model.n_channels,
            n_timepoints=cfg.model.n_timepoints,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout
        )
    elif model_type == 'eegnet':
        model = EEGNet(
            n_channels=cfg.model.n_channels,
            n_timepoints=cfg.model.n_timepoints,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout
        )
    elif model_type == 'attention':
        model = AttentionEEGNet(
            n_channels=cfg.model.n_channels,
            n_timepoints=cfg.model.n_timepoints,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout
        )
    else:
        # 原始模型
        model = EEGBinaryModel(
            n_channels=cfg.model.n_channels,
            n_timepoints=cfg.model.n_timepoints,
            d_model=cfg.model.d_model,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout
        )
    
    return model