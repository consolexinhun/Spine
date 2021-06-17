data_root_dir=./Verse
set -e
export CUDA_VISIBLE_DEVICES="3"
fold_ind=4
manual_seed=$(date +%s)
# echo "step 1: 创建粗分割的 H5...................................................................."
# python -u ./datasets/coarse_create_h5.py \
#     --data_root_dir=${data_root_dir} \
#     --fold_num=4 \
#     --train_num=172 \
#     --val_num=100 \
#     --total_num=172

echo "step 2: 粗分割阶段训练 DeepLab............................................................"
python -u train_coarse.py \
    --model=DeepLabv3_plus_skipconnection_3d \
    --fold_ind=${fold_ind} \
    --data_dir=${data_root_dir}/coarse \
    --no-pre_trained \
    --epochs=100 \
    --device=cuda:0 \
    --learning_rate=0.001 \
    --loss=CrossEntropyLoss \
    --seed=${manual_seed}

echo "step 3: 粗分割阶段用预训练的 DeepLab 训练 GCN.........................."
python -u train_coarse.py \
    --model=DeepLabv3_plus_gcn_skipconnection_3d \
    --fold_ind=${fold_ind} \
    --data_dir=${data_root_dir}/coarse \
    --pre_trained \
    --epochs=50 \
    --device=cuda:0 \
    --learning_rate=0.001 \
    --gcn_mode=2 \
    --ds_weight=0.3 \
    --loss=CrossEntropyLoss \
    --seed=${manual_seed}

echo "step 4: 抽取粗分割的概率图....................................................................................."
python -u coarse_semantic_feature.py \
    --device=cuda:0 \
    --fold_ind=${fold_ind} \
    --model=DeepLabv3_plus_gcn_skipconnection_3d \
    --data_dir=${data_root_dir}/coarse \
    --gcn_mode=2 \
    --ds_weight=0.3 \
    --loss=CrossEntropyLoss \
    --pre_trained \
    --learning_rate=0.001 \
    --seed=${manual_seed}


echo "拷贝粗分割文件到细化分割上"
cp -r ${data_root_dir}/coarse/*.npz ${data_root_dir}/fine/

echo "step 5: 创建细化分割数据集................................................................"
python -u ./datasets/fine_create_h5.py \
    --coarse_dir=${data_root_dir}/coarse \
    --fine_dir=${data_root_dir}/fine \
    --coarse_identifier=DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_Adam_lr_0.001_seed_${manual_seed}_pretrained

echo "step 6: 细化阶段训练 2D ResUNet....................................................................."
python -u train_fine.py \
    --fold_ind=${fold_ind} \
    --coarse_identifier=DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_Adam_lr_0.001_seed_${manual_seed}_pretrained \
    --model=ResidualUNet2D \
    --augment \
    --unary \
    --batch_size=1 \
    --data_dir=${data_root_dir}/fine \
    --device=cuda:0 \
    --epochs=50 \
    --manual_seed=${manual_seed}