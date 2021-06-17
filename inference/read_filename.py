import os, sys

data_dir = "/home/ubuntu/90_random_spine/intefence/test_nii/test1/MR"
a = sorted(os.listdir(data_dir))

b = [int(filename.replace("Case", "").replace(".nii.gz", "")) for filename in a]

b.sort()
print(b) # [1, 4, 28, 42, 44, 47, 50, 65, 70, 94, 95, 96, 99, 124, 126, 147, 157, 189, 195, 209]