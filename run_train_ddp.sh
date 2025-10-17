# 1. 给脚本添加执行权限
chmod +x examples/train_classification_ddp.sh

# 2. 修改脚本中的路径
#vim examples/train_classification_ddp.sh
# 修改 MODEL_PATH 和 TRAIN_FILE

# 3. 运行训练
bash examples/train_classification_ddp.sh