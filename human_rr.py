import scipy.io as sio
import numpy as np
import pickle
def cut2beats(n, sig_list, y):
    rr_mat = np.zeros((n, 1))
    lbls_mat = np.zeros((1, 2), dtype=int)
    for idx, sig in enumerate(sig_list):
        j = 0
        l_sig = sig.size
        while (j+1)*n < l_sig:
            rr_mat = np.hstack((rr_mat, sig[j*n:(j+1)*n]))
            j += 1
        lbls_mat_temp = np.zeros((j, 2), dtype=int)
        lbls_mat_temp[:, 0] = y[idx]
        lbls_mat = np.vstack((lbls_mat, lbls_mat_temp))
    rr_mat = rr_mat[:, 1:]
    lbls_mat = lbls_mat[1:, :]
    return rr_mat, lbls_mat

n = 128
filt_rr = sio.loadmat('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/filt_rr_cell.mat')
rr = filt_rr['filt_cell']
tags = rr[:, 0]
y = [int(tags[j].item()) for j,_ in enumerate(tags)]
sig_list = []
for i in range(rr.shape[0]):
    sig_list.append(rr[i, 2])
rr_mat, lbls_mat = cut2beats(n, sig_list, y)
rand_idx = np.random.choice(y, size=int(np.ceil(0.2*len(y))), replace=False)
rr_train, lbls_mat_train = rr_mat[:, ~np.isin(lbls_mat[:,0], rand_idx)], lbls_mat[~np.isin(lbls_mat[:,0], rand_idx), :]
rr_test, lbls_mat_test = rr_mat[:, np.isin(lbls_mat[:,0], rand_idx)], lbls_mat[np.isin(lbls_mat[:,0], rand_idx), :]
with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_train_data.pkl', 'wb') as f:
    pickle.dump([rr_train.T, lbls_mat_train], f)
with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_test_data.pkl', 'wb') as f:
    pickle.dump([rr_test.T, lbls_mat_test], f)


