import os
import pickle
import numpy as np
from tqdm import tqdm

barcodes_folder = ''
out_file = ''

def calculate_bins(barcode, bins=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]):
    num_in_bins = [0] * len(bins)
    for i in range(len(bins)):
        num_in_bins[i] = len([x for x in barcode if x[0] < bins[i] and x[1] > bins[i]])
    return num_in_bins

def calculate_barcodes_stats_by_layer(barcodes, dim=0):
    features = []

    new_barcodes = []
    for i in range(33):
        new_fts = []
        for j in range(i * 20, (i + 1) * 20):
            new_fts.extend(barcodes[j])
        new_barcodes.append(new_fts)
    barcodes = new_barcodes

    for lh in barcodes:
        lh_features = []
        h0_lens = np.array([x[1] - x[0] for x in lh if x[2] == 0])
        if len(h0_lens) == 0:
            lh_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            lh_features.append(np.sum(h0_lens))
            lh_features.append(np.median(h0_lens))
            lh_features.append(np.mean(h0_lens))
            lh_features.append(np.std(h0_lens))
            lh_features.append(np.max(h0_lens))
            lh_features.append(len(h0_lens))
            lens_normed = h0_lens / np.sum(h0_lens)
            lh_features.append(-np.sum(lens_normed * np.log(lens_normed)))
            lh_features.extend(calculate_bins([x for x in lh if x[2] == 0]))
        if dim == 1:
            h1_lens = np.array([x[1] - x[0] for x in lh if x[2] == 1])
            if len(h1_lens) == 0:
                lh_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                lh_features.append(np.sum(h1_lens))
                lh_features.append(np.median(h1_lens))
                lh_features.append(np.mean(h1_lens))
                lh_features.append(np.std(h1_lens))
                lh_features.append(np.max(h1_lens))
                lh_features.append(len(h1_lens))
                lens_normed = h1_lens / np.sum(h1_lens)
                lh_features.append(-np.sum(lens_normed * np.log(lens_normed)))
                lh_features.extend(calculate_bins([x for x in lh if x[2] == 1]))
        features.append(lh_features)
    return np.array(features)

def calculate_barcodes_stats(barcodes, dim=0):
    features = []
    for lh in barcodes:
        lh_features = []
        h0_lens = np.array([x[1] - x[0] for x in lh if x[2] == 0])
        if len(h0_lens) == 0:
            lh_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            lh_features.append(np.sum(h0_lens))
            lh_features.append(np.median(h0_lens))
            lh_features.append(np.mean(h0_lens))
            lh_features.append(np.std(h0_lens))
            lh_features.append(np.max(h0_lens))
            lh_features.append(len(h0_lens))
            lens_normed = h0_lens / np.sum(h0_lens)
            lh_features.append(-np.sum(lens_normed * np.log(lens_normed)))
            lh_features.extend(calculate_bins([x for x in lh if x[2] == 0]))
        if dim == 1:
            h1_lens = np.array([x[1] - x[0] for x in lh if x[2] == 1])
            if len(h1_lens) == 0:
                lh_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                lh_features.append(np.sum(h1_lens))
                lh_features.append(np.median(h1_lens))
                lh_features.append(np.mean(h1_lens))
                lh_features.append(np.std(h1_lens))
                lh_features.append(np.max(h1_lens))
                lh_features.append(len(h1_lens))
                lens_normed = h1_lens / np.sum(h1_lens)
                lh_features.append(-np.sum(lens_normed * np.log(lens_normed)))
                lh_features.extend(calculate_bins([x for x in lh if x[2] == 1]))
        features.append(lh_features)
    return np.array(features)

data_barcodes = {}
data_barcodes_layer = {}

files = os.listdir(barcodes_folder)
cnt = 0
for file in tqdm(files):
    with open(os.path.join(barcodes_folder, file), 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            data_barcodes[key] = {
                'h0': calculate_barcodes_stats(data[key]['barcodes'], dim=0),
                # 'embedding': data[key]['embeddings'].mean(axis=0),
                'mf': data[key]['label_MF'],
                'cc': data[key]['label_CC'],
                'bp': data[key]['label_BP']
            }
            data_barcodes_layer[key] = {
                'h0': calculate_barcodes_stats_by_layer(data[key]['barcodes'], dim=0),
                # 'embedding': data[key]['embeddings'].mean(axis=0),
                'mf': data[key]['label_MF'],
                'cc': data[key]['label_CC'],
                'bp': data[key]['label_BP']
            }
    cnt += 1
    if cnt % 100 == 0:
        with open(out_file, 'wb') as f:
            pickle.dump(data_barcodes, f)
        with open(out_file[-4] + '_layer.pkl', 'wb') as f:
            pickle.dump(data_barcodes_layer, f)

