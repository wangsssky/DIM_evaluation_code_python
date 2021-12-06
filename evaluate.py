import cv2
import os
import numpy as np

from tqdm import tqdm

from compute_sad_loss import compute_sad_loss
from compute_mse_loss import compute_mse_loss
from compute_gradient_loss import compute_gradient_loss
from compute_connectivity_error import compute_connectivity_error

if __name__ == '__main__':
    GT_DIR = './matting_evaluation/gt_alpha'
    TRI_DIR = './matting_evaluation/trimap'
    RE_DIR = './matting_evaluation/pred_alpha'
    DATA_TEST_LIST = './matting_evaluation/name_list.txt'

    fid = open(DATA_TEST_LIST, 'r')
    names = fid.readlines()

    sad = []
    mse = []
    grad = []
    conn = []
    for name in tqdm(names):
        try:
            imname = name.strip()

            pd = cv2.imread(os.path.join(RE_DIR, imname), cv2.IMREAD_GRAYSCALE)

            gt = cv2.imread(os.path.join(GT_DIR, imname), cv2.IMREAD_GRAYSCALE)
            tr = cv2.imread(os.path.join(TRI_DIR, imname), cv2.IMREAD_GRAYSCALE)

            sad.append(compute_sad_loss(pd, gt, tr))
            mse.append(compute_mse_loss(pd, gt, tr))
            grad.append(compute_gradient_loss(pd, gt, tr))
            conn.append(compute_connectivity_error(pd, gt, tr, 0.1))
        except Exception as e:
            pass

    SAD = np.mean(sad)
    MSE = np.mean(mse)
    GRAD = np.mean(grad)
    CONN = np.mean(conn)

    print('SAD: {:.4f}, MSE: {:.4f}, Grad: {:.4f}, Conn: {:.4f} \n'.format(SAD, MSE, GRAD, CONN))
