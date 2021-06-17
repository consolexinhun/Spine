data_root_dir=/home/ubuntu/projects/Spine/CBAM
set -e
export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES="0"

# echo "step 1: 创建粗分割的 H5...................................................................."
# python -u ./inference/coarse_create_h5.py \
#     --data_root_dir=${data_root_dir}/test_nii/test1_CBAM_2
# 会创建 h5 和 npz 文件

# echo "step 2: 抽取粗分割的概率图....................................................................................."
# python -u inference/coarse_semantic_feature.py \
#     --device=cuda:0 \
#     --model=DeepLabv3_plus_gcn_skipconnection_3d \
#     --data_dir=${data_root_dir}/test_nii/test1/coarse \
#     --gcn_mode=2 \
#     --ds_weight=0.3 \
#     --loss=CrossEntropyLoss \
#     --pre_trained \
#     --model_path=/home/ubuntu/projects/Spine/Spine_CBAM_Deeplab_behind/Verse/coarse

## 粗分割下产生 out 文件夹


echo "step 3: 测试 粗细分割的结果................................................................................................................"
python -u inference/test_coarse_fine.py \
    --device=cuda:0 \
    --data_dir=${data_root_dir}/inference/test_nii/test1/ \
    --pre_trained \
    --seed=1623804195 \
    --coarse_dir=${data_root_dir}/Verse/coarse/model/fold1 \
    --fine_dir=${data_root_dir}/Verse/fine/model/fold1_DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_Adam_lr_0.001_seed_1623804195_pretrained
## 产生 coarse_fine 结果，这个就是推断的结果
