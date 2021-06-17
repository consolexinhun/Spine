import sys
sys.path.append('networks')
import h5py
import numpy as np
import os
import nibabel as nib
from skimage import transform
import matplotlib.pyplot as plt
from scipy import ndimage
from argparse import ArgumentParser

if __name__ == '__main__':

    height = 256
    width = 512

    mean = 466.0
    std = 379.0

    parser = ArgumentParser()
    parser.add_argument("--coarse_dir", type=str, default='/public/pangshumao/data/five-fold/coarse',
                        help="coarse dir")
    parser.add_argument("--fine_dir", type=str, default='/public/pangshumao/data/five-fold/fine',
                        help="fine dir")
    parser.add_argument("--unary_dir", type=str)
    args = parser.parse_args()

    coarseDir = args.coarse_dir
    fineDir = args.fine_dir

    mrDir = os.path.join(coarseDir, 'MR')
    foldIndData = np.load(os.path.join(fineDir, 'split_ind_fold1.npz'))
    unaryDir = args.unary_dir
    outDir = os.path.join(fineDir, 'h5py')

    train_ind = []
    val_ind = []
    test_ind = foldIndData['test_ind']

    # calculate the slices number
    train_slice_num = 0
    val_slice_num = 0
    test_slice_num = 0

    for i in test_ind:
        temp = nib.load(os.path.join(mrDir, f'Case{i}.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
        test_slice_num += temp.shape[0]

    try:
        f = h5py.File(os.path.join(outDir, 'fold1_data.h5'), 'w')

        g_train = f.create_group('train')
        g_val = f.create_group('val')
        g_test = f.create_group('test')
        
        # For test data
        flag = True
        for i in test_ind:
            print('processing test fold1, case%d' % (i))
            mr = nib.load(os.path.join(mrDir, f'Case{i}.nii.gz')).get_data().transpose(2, 0,
                                                                                                    1)  # [d, h, w]
            mr -= mean
            mr /= std

            unary = np.load(os.path.join(unaryDir, f'Case{i}_logit.npz'))['logit']  # [1, 20, d, 128, 256]

            d, h, w = mr.shape
            start_h = int(h / 4.)
            end_h = -int(h / 4.)
            mr = mr[:, start_h:end_h, :]  # [d, h/2, w]

            # resize the data to the same shape
            mr = transform.resize(mr.astype(np.float), (d, height, width), order=3,
                                    mode='constant')  # [d, height, width]
            mr = np.reshape(mr, (d, 1, height, width))  # [d, 1, height, width]

            unary = np.squeeze(unary)
            unary = unary.transpose((1, 0, 2, 3))  # [d, 20, height, width]

            if flag:
                flag = False
                g_test.create_dataset('mr', data=mr,
                                        maxshape=(test_slice_num, mr.shape[1], mr.shape[2], mr.shape[3]),
                                        chunks=(1, mr.shape[1], mr.shape[2], mr.shape[3]))
               
                g_test.create_dataset('unary', data=unary,
                                        maxshape=(test_slice_num, unary.shape[1], unary.shape[2], unary.shape[3]),
                                        chunks=(1, unary.shape[1], unary.shape[2], unary.shape[3]))
            else:
                g_test['mr'].resize(g_test['mr'].shape[0] + mr.shape[0], axis=0)
                g_test['mr'][-mr.shape[0]:] = mr

                g_test['unary'].resize(g_test['unary'].shape[0] + unary.shape[0], axis=0)
                g_test['unary'][-unary.shape[0]:] = unary

    finally:
        f.close()
    print('Job done!!!')