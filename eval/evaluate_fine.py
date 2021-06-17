import numpy as np
import os
from scipy.stats import ttest_rel as ttest

if __name__ == '__main__':
    dataDir = '/public/pangshumao/data/five-fold'

    subDir1 = 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_Adam_lr_0.001_pretrained'
    subDir2 = 'ResidualUNet2D_Adam_lr_0.0001_weight_decay_0.0001_augment'

    dice1 = np.zeros((215))
    dice2 = np.zeros((215))
    for i in range(1, 6):
        data1 = np.load(os.path.join(dataDir, 'coarse', 'out', 'fold' + str(i), subDir1, 'eval_scores.npz'))['dice']
        data2 = np.load(os.path.join(dataDir, 'fine', 'out', 'fold' + str(i), subDir2, 'eval_scores.npz'))['crf_dices']
        dice1[(i - 1) * 43 : i * 43] = data1
        dice2[(i - 1) * 43: i * 43] = data2
    print(np.mean(dice1))
    print(np.mean(dice2))
    s, p = ttest(dice1, dice2)
    print('s = ', s)
    print('p = ', p)
    count1 = np.sum(dice1 < 0.75)
    count2 = np.sum(dice2 < 0.75)
    print(count1)
    print(count2)
    sort1 = np.sort(dice1)
    sort2 = np.sort(dice2)
    pass


