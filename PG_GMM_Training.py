import glob
import numpy as np
import matplotlib.pyplot as plt

from common import emgm
from get_pg import Get_PG

TD_path = "F:/code/PG-GMM_TrainingCode/Kodak24/"
step = 3
delta = 0.002
win = 15
ps = 8
nlsp = 10
cls_num = 32
# read natural clean images
im_dir = glob.glob(TD_path + '*.tif')
im_num = 1# len(im_dir)
X = []
X0 = []
for i in range(im_num):
    print("%d/%d: " % (i, im_num) + im_dir[i])
    im = plt.imread(im_dir[i]).astype(np.float32)
    im = im / 255
    Px, Px0 = Get_PG(im, win, ps, nlsp, step, delta)
    X0.append(Px0)
    X.append(Px)
X0 = np.concatenate(X0,axis=1)
X = np.concatenate(X,axis=1)

[model, llh, cls_idx] = emgm(X, np.array(cls_num), nlsp)
