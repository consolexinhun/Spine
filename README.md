# 安装环境

- pytorch
- torchvision
- scipy
- tensorboardX
- numpy
- opencv-python
- matplotlib
- networkx
- h5py
- scikit-iamge
- medpy
- nibabel
- torchsummary

# 数据集

数据集在 219 机器的 /mnt/ssd/dengyang/Verse

拷贝过来直接放到 项目根目录下

# 运行

1. 指定数据集的目录 

`data_root_dir=/home/ubuntu/03_SpineParseNet/Verse`

2. 设置可见显卡 看哪张卡空了用哪张

`CUDA_VISIBLE_DEVICES="1"` 

3. 创建粗分割数据划分

`python -u ./datasets/coarse_create_h5.py --data_root_dir=${data_root_dir}`

4. 训练粗分割 

`python -u train_coarse.py --model=DeepLabv3_plus_skipconnection_3d --fold_ind=1 --data_dir=${data_root_dir}/coarse --no-pre_trained --epochs=100 --device=cuda:0 --learning_rate=0.001 --loss=CrossEntropyLoss`

5. GCN 粗分割 

`python -u train_coarse.py --model=DeepLabv3_plus_gcn_skipconnection_3d --fold_ind=1 --data_dir=${data_root_dir}/coarse --pre_trained --epochs=50 --device=cuda:0 --learning_rate=0.001 --gcn_mode=2 --ds_weight=0.3 --loss=CrossEntropyLoss`

6. 抽取粗分割特征

`python -u coarse_semantic_feature.py --device=cuda:0 --fold_ind=${fold_ind} --model=DeepLabv3_plus_gcn_skipconnection_3d --data_dir=${data_root_dir}/coarse --gcn_mode=2 --ds_weight=0.3 --loss=CrossEntropyLoss --pre_trained`

7. 创建细化分割数据划分

`python -u ./datasets/fine_create_h5.py --coarse_dir=${data_root_dir}/coarse --fine_dir=${data_root_dir}/fine`

8. 训练细化分割

`python -u train_fine.py --fold_ind=1 --data_dir=${data_root_dir}/fine --device=cuda:0`

9. 测试（暂时还用不了）

`python -u test_coarse_fine.py --device=cuda:0 --fold_ind=${fold_ind} --data_dir=${data_root_dir}`


