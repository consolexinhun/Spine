# 安装环境

+ pytorch
+ torchvision
+ scipy
+ tensorboardX
+ numpy
+ opencv-python
+ matplotlib
+ networkx

# fix bug

0、datasets/coarse_create_h5
line 35: train_num 168
line 38: val_num 4
line 41: total_num 172

1、datasets/coarse_create_h5
line 7:把 matplotlib 的 TKAgg注释

networks/losses.py
line 7:把 matplotlib 的 TKAgg注释


2、datasets/coarse_h5
line 74: 注释
line 81：注释
line 83：改为 `return training_data_loader, val_data_loader, None, f`

datasets/fine_h5

line 69: 注释
line 76：注释
line 78：改为 `return training_data_loader, val_data_loader, None, f`


3、coarse_semantic_feature

line 139：254 改为 173

line 326：把 `conf['trainer']['log_after_iters'] = args.log_after_iters` 注释掉就行了
反正这个参数又没有用到


4、fine 里面没有划分数据，需要从 coarse 里面拷贝
（执行完 main.sh 第一步之后）

进入到 fine 目录

`cp ../coarse/*.npz ./`


