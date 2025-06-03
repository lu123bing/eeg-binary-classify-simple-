# 检查所有保存的checkpoint文件
import os
checkpoint_dir = "你的checkpoint目录"
for file in os.listdir(checkpoint_dir):
    if file.endswith('.ckpt'):
        print(f"Checkpoint文件: {file}")
        # 看文件名中是否包含准确率信息