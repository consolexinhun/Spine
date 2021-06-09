从原始仓库 clone 下来，一堆 bug 需要解决


# fix bug

## -1. main.sh

step 1:

python -u ./datasets/coarse_create_h5.py --data_root_dir=${data_root_dir} --fold_num=1

## 0、datasets/coarse_create_h5

line 7:把 matplotlib 的 TKAgg注释

line 35: train_num 168

line 38: val_num 4

line 41: total_num 172

## 1、networks/losses.py


line 7:把 matplotlib 的 TKAgg注释


## 2、datasets/coarse_h5
line 74: 注释

line 81 82：注释

line 83：改为 `return training_data_loader, val_data_loader, None, f`

datasets/fine_h5

line 69: 注释

line 76, 77：注释

line 78：改为 `return training_data_loader, val_data_loader, None, f`


3、coarse_semantic_feature

line 439：216 改为 173

line 326：把 `conf['trainer']['log_after_iters'] = args.log_after_iters` 注释掉就行了
反正这个参数又没有用到


4、fine 里面没有划分数据，需要从 coarse 里面拷贝
（执行完 main.sh 第一步之后）

进入到 fine 目录

`cp ../coarse/*.npz ./`


5、datasets/fine_create_h5

line 128 ： 6 改成 2 （5 折 变为 1 折）