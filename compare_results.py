import os, glob, math
import numpy as np
from scipy.io import loadmat, savemat
import cv2

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(SCRIPT_PATH,'dataset')

# Read ground truth
def ground_truth(name):
    return os.path.join(SCRIPT_PATH, 'dataset', 'ground-truth','%s.exr' % name)

def results(name):
    return os.path.join(SCRIPT_PATH, 'results', name, 'depth.mat')

def sphere_cnn(name):
    return os.path.join(SCRIPT_PATH, 'sphere-cnn', name, 'predict_depth.mat')

for file in glob.glob(os.path.join(dataset_dir, "*.png")):
    name = os.path.basename(file).replace('.png','')
    direct = loadmat(sphere_cnn(name))['data_obj']
    ours = cv2.resize(loadmat(results(name))['data_obj']*50, direct.shape, cv2.INTER_AREA)
    ground_t = cv2.resize(cv2.imread(ground_truth(name), cv2.IMREAD_UNCHANGED)[...,0], direct.shape, cv2.INTER_AREA)
    direct = direct.T
    im = ours.ravel()
    sp = direct.ravel()
    gt = ground_t.ravel()
    n = float(len(im))
    assert(im.shape == gt.shape and im.shape == sp.shape and n > 0)
    sum_di_ours = 0
    sum_di_direct = 0
    sum_di_sq_ours = 0
    sum_di_sq_direct = 0
    for est_i, dir_i, gt_i in zip(im, sp, gt):
        if est_i > 0:
            di_ours = math.log(est_i) - math.log(gt_i)
            di_direct = math.log(dir_i) - math.log(gt_i)
            sum_di_ours += di_ours
            sum_di_sq_ours += di_ours*di_ours
            sum_di_direct += di_direct
            sum_di_sq_direct += di_direct*di_direct
    error_ours =  (1/n)*sum_di_sq_ours - 1/(n**2)*(sum_di_ours**2)
    error_direct =  (1/n)*sum_di_sq_direct - 1/(n**2)*(sum_di_direct**2)
    print(name, error_ours, error_direct)
