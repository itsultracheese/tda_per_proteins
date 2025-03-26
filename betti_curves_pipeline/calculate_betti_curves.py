import os
import pickle
import numpy as np
from tqdm import tqdm
from gtda.diagrams import BettiCurve

train_folder = ''
test_folder = ''
out_train_folder = ''
out_test_folder = ''

files_train = os.listdir(train_folder)
files_test = os.listdir(test_folder)


n_bins = 80

for file in tqdm(files_train, desc='loading train'):
    with open(os.path.join(train_folder, file), 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            barcodes = data[key]['barcodes']
            labels_mf = data[key]['label_MF']
            labels_cc = data[key]['label_CC']
            labels_bp = data[key]['label_BP']

    bc = BettiCurve(n_bins=n_bins)
    # pad to have the same bins
    placeholder = np.array([1.0, 1.0, 0.0])
    barcodes = np.array(barcodes)
    t = np.tile(placeholder, (barcodes.shape[0], 1))
    barcodes = np.concatenate((barcodes, t.reshape(-1, 1, 3)), axis=1)
    curves = bc.fit_transform(barcodes)
    curves = curves[:, 0, :]
    res = {
        'betticurves': curves,
        'label_MF': np.array(labels_mf),
        'label_CC': np.array(labels_cc),
        'label_BP': np.array(labels_bp)
    }
    with open(os.path.join(out_train_folder, f'bins_{n_bins}' + file), 'wb') as w:
        pickle.dump(res, w)

for file in tqdm(files_test, desc='loading test'):
    with open(os.path.join(test_folder, file), 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            barcodes = data[key]['barcodes']
            labels_mf = data[key]['label_MF']
            labels_cc = data[key]['label_CC']
            labels_bp = data[key]['label_BP']

    bc = BettiCurve(n_bins=n_bins)
    # pad to have the same bins
    placeholder = np.array([1.0, 1.0, 0.0])
    barcodes = np.array(barcodes)
    t = np.tile(placeholder, (barcodes.shape[0], 1))
    barcodes = np.concatenate((barcodes, t.reshape(-1, 1, 3)), axis=1)
    curves = bc.fit_transform(barcodes)
    curves = curves[:, 0, :]
    res = {
        'betticurves': curves,
        'label_MF': np.array(labels_mf),
        'label_CC': np.array(labels_cc),
        'label_BP': np.array(labels_bp)
    }
    with open(os.path.join(out_test_folder, f'bins_{n_bins}' + file), 'wb') as w:
        pickle.dump(res, w)