import h5py
import numpy as np
import os
import nibabel as nib
from skimage import transform
import math
from scipy import ndimage
from argparse import ArgumentParser
import random

def compute_distance_weight_matrix(mask, alpha=1, beta=8, omega=2):
    mask = np.asarray(mask)
    distance_to_border = ndimage.distance_transform_edt(mask > 0) + ndimage.distance_transform_edt(mask == 0)
    weights = alpha + beta * np.exp(-(distance_to_border ** 2 / omega ** 2))
    return np.asarray(weights, dtype='float32')

if __name__ == '__main__':
    mean = 466.0
    std = 379.0

    depth = 15
    height = 128
    width = 256
    class_num = 20
    parser = ArgumentParser()

    parser.add_argument("--data_root_dir", type=str, default='data',
                        help="the absolute directory of the data.")

    
    
    args = parser.parse_args()

    data_root_dir = args.data_root_dir
    

    # 原图
    mrDir = os.path.join(data_root_dir, 'test_nii/test1/coarse/MR')
    # 数据
    foldIndDir = os.path.join(data_root_dir, 'test_nii/test1/coarse')
    # h5
    outDir = os.path.join(data_root_dir, 'test_nii/test1/coarse/h5py')
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    test_ind = np.array([1, 4, 28, 42, 44, 47, 50, 65, 70, 94, 95, 96, 99, 124, 126, 147, 157, 189, 195, 209])
    test_num = len(test_ind)

    # 保存 npz
    np.savez(os.path.join(foldIndDir, 'split_ind_fold1.npz'), 
        train_ind=[],val_ind=[], test_ind=test_ind)

    f = h5py.File(os.path.join(outDir, 'data_fold1.h5'), 'w')
    g_train = f.create_group('train')
    g_val = f.create_group('val')
    g_test = f.create_group('test')

    # For test data
    flag = True
    for j in test_ind:
        print(f"正在读取文件 :Case{j}.nii.gz")
        mr = nib.load(os.path.join(mrDir, f'Case{j}.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]

        d, h, w = mr.shape
        shape = np.array([[d, h, w]])

        start_h = int(h / 4.)
        end_h = -int(h / 4.)
        mr = mr[:, start_h:end_h, :]  # [d, h/2, w]

        mr_resize = transform.resize(mr.astype(np.float), (d, height, width), order=3,
                                        mode='constant').astype(np.float32)  # [d, height, width]

        mr_out = np.zeros((1, 1, depth, height, width), dtype=np.float32)

        delata = depth - d
        start_d = random.randint(0,delata)
        end_d = start_d+d

        mr_out[0, 0, start_d: end_d, :, :] = mr_resize
        mr_out -= mean
        mr_out /= std


        if flag:
            flag = False
            g_test.create_dataset('mr', data=mr_out,
                maxshape=(test_num, mr_out.shape[1], mr_out.shape[2], mr_out.shape[3],mr_out.shape[4]),
                chunks=(1, mr_out.shape[1], mr_out.shape[2], mr_out.shape[3], mr_out.shape[4]))
            
            g_test.create_dataset('shape', data=shape,
                maxshape=(test_num, shape.shape[1]),
                chunks=(1, shape.shape[1]))
        else:
            g_test['mr'].resize(g_test['mr'].shape[0] + mr_out.shape[0], axis=0)
            g_test['mr'][-mr_out.shape[0]:] = mr_out
            
            g_test['shape'].resize(g_test['shape'].shape[0] + shape.shape[0], axis=0)
            g_test['shape'][-shape.shape[0]:] = shape
    f.close()