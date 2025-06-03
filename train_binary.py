import hydra
from omegaconf import DictConfig
import torch
from models import create_binary_model
from pl_models import BinaryExtractorModel
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pl_datamodule import EEGBinaryDataModule
import os
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../cfgs", config_name="binary_config", version_base="1.3")
def train_binary_ext(cfg: DictConfig) -> None:
    """训练二分类提取器 - 优化版"""
    
    print("🚀 开始训练EEG二分类提取器 (优化版)")
    print(f"🔍 模型: {cfg.model.model_type}")
    
    # 🔥 系统优化设置
    if torch.cuda.is_available():
        # print(f"🔍 系统检查:")
        # print(f"   CUDA可用: True")
        # print(f"   PyTorch版本: {torch.__version__}")
        # print(f"   GPU: {torch.cuda.get_device_name(0)}")
        # print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 🔥 CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('medium')
        
        # 🔥 清理GPU缓存
        torch.cuda.empty_cache()
    
    # 🔥 减少worker数量，避免内存瓶颈
    if cfg.train.num_workers > 4:
        cfg.train.num_workers = 4
        print(f"⚠️ 限制worker数量为4以提升性能")
    
    # 设置随机种子
    pl.seed_everything(cfg.seed)
    
    # 🔥 简化交叉验证 - 减少fold数量用于快速测试
    n_folds = min(cfg.train.valid_method, 10)  # 🔥 最多5折
    print(f"📊 简化交叉验证: {n_folds}折 (原{cfg.train.valid_method}折)")
    
    n_per = cfg.data.n_subs // n_folds
    remainder = cfg.data.n_subs % n_folds
    
    print(f"📊 {n_folds}折交叉验证配置:")
    print(f"   数据集: {cfg.data.dataset_name}")
    print(f"   总被试数: {cfg.data.n_subs}")
    print(f"   交叉验证: {n_folds}折")
    print(f"   每折被试数: {n_per}")
    if remainder > 0:
        print(f"   最后一折额外被试数: {remainder}")
    print(f"   输入维度: ({cfg.model.n_channels}, {cfg.model.n_timepoints})")
    print(f"   批次大小: {cfg.data.batch_size}")
    print(f"   最大轮次: {cfg.train.max_epochs}")
    print(f"   学习率: {cfg.train.lr}")
    
    fold_results = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # 🔥 交叉验证主循环
    for fold in range(n_folds):
        print(f"\n🔄 开始第 {fold+1}/{n_folds} 折训练...")
        
        # 🔥 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 数据划分
        start_idx = fold * n_per
        if fold == n_folds - 1:
            end_idx = cfg.data.n_subs
        else:
            end_idx = (fold + 1) * n_per
            
        val_subs = np.arange(start_idx, end_idx)
        train_subs = np.concatenate([
            np.arange(0, start_idx),
            np.arange(end_idx, cfg.data.n_subs)
        ])
        
        print(f"   验证被试: {len(val_subs)} 人 (索引: {start_idx}-{end_idx-1})")
        print(f"   训练被试: {len(train_subs)} 人")
        print(f"   验证被试ID: {val_subs.tolist()}")
        
        # 设置检查点目录
        cp_dir = os.path.join(cfg.log.cp_dir, 'binary_' + cfg.data.dataset_name +'_' + cfg.model.model_type, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        
        # 🔥 简化wandb设置
        wandb_logger = WandbLogger(
            name=f"binary_fold{fold+1}_r{cfg.log.run}",
            project=cfg.log.proj_name,
            log_model=False,  # 🔥 不保存模型到wandb，节省时间
            offline=True  # 🔥 离线模式，避免网络延迟
        )

        # 🔥 回调优化
        checkpoint_callback = ModelCheckpoint(
            monitor="ext/val/acc", 
            mode="max", 
            verbose=False,  # 🔥 减少日志输出
            dirpath=cp_dir, 
            filename=f'fold{fold+1}_best',
            save_top_k=1,
            save_last=False  # 🔥 不保存最后一个模型
        )
        
        earlyStopping_callback = EarlyStopping(
            monitor="ext/val/acc", 
            mode="max", 
            patience=min(cfg.train.patience, 20),  # 🔥 减少patience
            verbose=False
        )

        # 视频设置
        n_vids = 24 if cfg.data.dataset_name == 'FACED' else cfg.data.n_vids
        train_vids = np.arange(n_vids)
        val_vids = np.arange(n_vids)

        # 🔥 创建数据模块 - 减少worker
        dm = EEGBinaryDataModule(
            cfg.data, train_subs.tolist(), val_subs.tolist(), 
            train_vids, val_vids, False, 
            min(cfg.train.num_workers, 2)  # 🔥 最多2个worker
        )

        # 创建模型
        print(f"🔧 Fold {fold+1} 模型信息:")
        model = create_binary_model(cfg)
        
        # 🔥 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        print(f"   参数量: {total_params:,}")
        print(f"   模型大小: {model_size_mb:.2f} MB")
        
        Extractor = BinaryExtractorModel(model, cfg.train)
        
        # 🔥 训练器优化
        trainer = pl.Trainer(
            logger=wandb_logger, 
            callbacks=[checkpoint_callback, earlyStopping_callback],
            max_epochs=min(cfg.train.max_epochs, 100),  # 🔥 限制最大轮次
            min_epochs=cfg.train.min_epochs, 
            accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
            devices=1,  # 🔥 只用一个GPU
            precision='32-true',  # 
            limit_val_batches=1.0,  # 🔥 使用所有验证批次
            val_check_interval=1.0,  # 🔥 每半个epoch验证一次
            enable_progress_bar=True,
            enable_model_summary=False,  # 🔥 不显示模型摘要
            deterministic=False,
            gradient_clip_val=1.0,  # 🔥 梯度裁剪
            log_every_n_steps=100,  # 🔥 减少日志频率
            sync_batchnorm=False,  # 🔥 不同步BN
            fast_dev_run=False
        )
        
        print(f"🎯 开始Fold {fold+1}训练...")
        try:
            trainer.fit(Extractor, dm)
            
            # 🔥 优先使用模型自己记录的最佳指标
            if hasattr(Extractor, 'get_best_metrics'):
                best_metrics = Extractor.get_best_metrics()
                if best_metrics:
                    print(f"\n📊 使用模型记录的最佳指标:")
                    print(f"   最佳epoch: {best_metrics.get('epoch', 'unknown')}")
                    print(f"   验证准确率: {best_metrics.get('val_acc', 0.0):.4f}")
                    print(f"   验证损失: {best_metrics.get('val_loss', 0.0):.4f}")
                    print(f"   最优阈值: {best_metrics.get('optimal_threshold', 0.5):.4f}")
                    
                    best_val_acc = best_metrics.get('val_acc', 0.0)
                    best_val_loss = best_metrics.get('val_loss', 0.0)
                    
                    # 🔥 收集结果
                    if best_val_acc > 0:
                        fold_results['val_acc'].append(best_val_acc)
                    if best_val_loss < float('inf'):
                        fold_results['val_loss'].append(best_val_loss)
                    
                    print(f"✅ Fold {fold+1} 真实最佳验证准确率: {best_val_acc:.4f}")
                    
                    # 🔥 使用当前训练好的模型进行分析（它已经是经过完整训练的）
                    print(f"\n🔍 Fold {fold+1} 预测分析 (完整训练模型):")
                    Extractor.eval()
                    val_loader = dm.val_dataloader()
                    val_predictions, val_targets, val_probs, val_logits = analyze_predictions(
                        Extractor, val_loader, "验证集"
                    )
                    
                    # 继续下一个fold
                    continue
            
            # 🔥 如果模型没有记录，使用原来的逻辑
            # ... existing checkpoint logic ...
            # 🔥 获取最佳结果
            best_val_acc = 0.0
            best_train_acc = 0.0
            best_val_loss = float('inf')
            best_train_loss = float('inf')
            
            # 🔥 方法1：从ModelCheckpoint获取最佳验证准确率
            if checkpoint_callback.best_model_path:
                best_val_acc = float(checkpoint_callback.best_model_score)
                print(f"📊 最佳验证准确率: {best_val_acc:.4f}")
                
                # 🔥 加载最佳模型来获取完整指标
                try:
                    best_model = BinaryExtractorModel.load_from_checkpoint(
                        checkpoint_callback.best_model_path,
                        model=model,
                        cfg_train=cfg.train
                    )
                    
                    # 🔥 重新在验证集上评估
                    best_model.eval()
                    val_results = trainer.validate(best_model, dm, verbose=False)
                    
                    if val_results:
                        best_val_loss = float(trainer.callback_metrics.get('ext/val/loss', best_val_loss))
                    
                    # 🔥 在训练集上评估（获取对应的训练指标）
                    train_results = trainer.validate(best_model, dm.train_dataloader(), verbose=False)
                    if train_results:
                        # 注意：这里实际上是在训练数据上做验证，所以指标名可能不同
                        train_metrics = trainer.callback_metrics
                        for key, value in train_metrics.items():
                            if 'acc' in key:
                                best_train_acc = float(value)
                            elif 'loss' in key:
                                best_train_loss = float(value)
                        
                except Exception as e:
                    print(f"⚠️ 加载最佳模型失败: {e}")
                    # 使用当前模型的结果
                    best_val_acc = float(trainer.callback_metrics.get('ext/val/acc', 0.0))
                    best_val_loss = float(trainer.callback_metrics.get('ext/val/loss', 0.0))
                    best_train_acc = float(trainer.callback_metrics.get('ext/train/acc', 0.0))
                    best_train_loss = float(trainer.callback_metrics.get('ext/train/loss', 0.0))
            
            else:
                # 🔥 如果没有检查点，使用最后的结果
                metrics = trainer.callback_metrics
                best_val_acc = float(metrics.get('ext/val/acc', 0.0))
                best_val_loss = float(metrics.get('ext/val/loss', 0.0))
                best_train_acc = float(metrics.get('ext/train/acc', 0.0))
                best_train_loss = float(metrics.get('ext/train/loss', 0.0))
            
            # 🔥 显示最佳结果
            print(f"\n📊 Fold {fold+1} 最佳结果:")
            print(f"   📈 训练: 准确率={best_train_acc:.4f}, 损失={best_train_loss:.4f}")
            print(f"   📊 验证: 准确率={best_val_acc:.4f}, 损失={best_val_loss:.4f}")
            
            # 🔥 收集最佳结果
            if best_val_acc > 0:
                fold_results['val_acc'].append(best_val_acc)
            if best_val_loss < float('inf'):
                fold_results['val_loss'].append(best_val_loss)
            if best_train_acc > 0:
                fold_results['train_acc'].append(best_train_acc)
            if best_train_loss < float('inf'):
                fold_results['train_loss'].append(best_train_loss)
            
            print(f"✅ Fold {fold+1} 最佳验证准确率: {best_val_acc:.4f}")
            
            # 🔥 使用最佳模型进行预测分析
            if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
                try:
                    print(f"\n🔍 Fold {fold+1} 预测分析 (最佳模型):")
                    best_model = BinaryExtractorModel.load_from_checkpoint(
                        checkpoint_callback.best_model_path,
                        model=model,
                        cfg_train=cfg.train
                    )
                    best_model.eval()
                    
                    # 分析验证集预测
                    val_loader = dm.val_dataloader()
                    val_predictions, val_targets, val_probs, val_logits = analyze_predictions(
                        best_model, val_loader, "验证集"
                    )
                    
                except Exception as e:
                    print(f"⚠️ 最佳模型分析失败: {e}")
            else:
                # 使用当前模型
                print(f"\n🔍 Fold {fold+1} 预测分析 (当前模型):")
                Extractor.eval()
                val_loader = dm.val_dataloader()
                val_predictions, val_targets, val_probs, val_logits = analyze_predictions(
                    Extractor, val_loader, "验证集"
                )
            
        except Exception as e:
            print(f"❌ Fold {fold+1} 训练失败: {e}")
            import traceback
            traceback.print_exc()

        # 🔥 清理wandb
        try:
            wandb.finish()
        except:
            pass
        
        # 🔥 测试模式只跑一折
        if cfg.train.iftest:
            print("🧪 测试模式，只运行一折")
            break
    
    # 计算总体结果
    if len(fold_results['val_acc']) > 0:
        mean_val_acc = np.mean(fold_results['val_acc'])
        std_val_acc = np.std(fold_results['val_acc'])
        
        print(f"\n🎉 交叉验证结果:")
        print(f"   验证准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
        print(f"   成功fold数: {len(fold_results['val_acc'])}/{n_folds}")
    else:
        print(f"\n❌ 没有成功的fold")

def analyze_predictions(model, dataloader, dataset_name):
    """分析模型预测结果"""
    import torch
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []
    all_logits = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= 10:  # 只分析前10个batch，避免太慢
                break
                
            x = x.to(model.device)
            y = y.to(model.device)
            
            # 获取模型输出
            logits = model.forward(x).squeeze(1)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # 转换为numpy数组
    all_logits = np.array(all_logits)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    print(f"   📊 {dataset_name} 预测分布:")
    print(f"      样本数: {len(all_targets)}")
    print(f"      真实标签分布: 负样本={np.sum(all_targets==0)}, 正样本={np.sum(all_targets==1)}")
    print(f"      预测标签分布: 负样本={np.sum(all_predictions==0)}, 正样本={np.sum(all_predictions==1)}")
    
    # print(f"   📈 Logits统计:")
    # print(f"      范围: [{all_logits.min():.3f}, {all_logits.max():.3f}]")
    # print(f"      均值: {all_logits.mean():.3f} ± {all_logits.std():.3f}")
    
    print(f"   📈 概率统计:范围: [{all_probs.min():.3f}, {all_probs.max():.3f}],均值: {all_probs.mean():.3f} ± {all_probs.std():.3f}")
    # print(f"      范围: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
    # print(f"      均值: {all_probs.mean():.3f} ± {all_probs.std():.3f}")
    
    # 按类别分析
    if len(np.unique(all_targets)) > 1:
        pos_indices = all_targets == 1
        neg_indices = all_targets == 0
        
        if np.any(pos_indices):
            pos_probs = all_probs[pos_indices]
            pos_logits = all_logits[pos_indices]
            print(f"   📊 正样本统计:")
            print(f"      概率: {pos_probs.mean():.3f} ± {pos_probs.std():.3f}")
            print(f"      logits: {pos_logits.mean():.3f} ± {pos_logits.std():.3f}")
        
        if np.any(neg_indices):
            neg_probs = all_probs[neg_indices]
            neg_logits = all_logits[neg_indices]
            # print(f"   📊 负样本统计:")
            # print(f"      概率: {neg_probs.mean():.3f} ± {neg_probs.std():.3f}")
            # print(f"      logits: {neg_logits.mean():.3f} ± {neg_logits.std():.3f}")
        
        # 计算分离度
        if np.any(pos_indices) and np.any(neg_indices):
            prob_separation = abs(pos_probs.mean() - neg_probs.mean())
            logit_separation = abs(pos_logits.mean() - neg_logits.mean())
            print(f"   🎯 类别分离度:")
            print(f"      概率分离: {prob_separation:.3f}")
            print(f"      logits分离: {logit_separation:.3f}")
            
            if prob_separation < 0.1:
                print(f"      ⚠️ 概率分离度过小，模型可能没有学会区分")
            else:
                print(f"      ✅ 概率分离度良好")
    
    # 混淆矩阵
    if len(np.unique(all_targets)) > 1 and len(np.unique(all_predictions)) > 1:
        cm = confusion_matrix(all_targets, all_predictions)
        print(f"   📊 混淆矩阵:")
        print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"      FN={cm[1,0]}, TP={cm[1,1]}")
    
    return all_predictions, all_targets, all_probs, all_logits

if __name__ == "__main__":
    train_binary_ext()