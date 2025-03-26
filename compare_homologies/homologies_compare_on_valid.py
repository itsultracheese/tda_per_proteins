import time
import json
import pickle
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from scripts.helper_scripts.f1_max_score import count_f1_max

valid_stats_file = ''

with open(valid_stats_file, 'rb') as f:
    data_barcodes = pickle.load(f)

features_h0 = []
features_bip = []
features_h1 = []
embeddings = []
labels_mf = []
labels_cc = []
labels_bp = []
for key in tqdm(data_barcodes.keys()):
    features_h0.append(data_barcodes[key]['h0'])
    features_bip.append(data_barcodes[key]['bip'])
    features_h1.append(data_barcodes[key]['h1'])
    embeddings.append(data_barcodes[key]['embedding'])
    labels_mf.append(data_barcodes[key]['mf'])
    labels_cc.append(data_barcodes[key]['cc'])
    labels_bp.append(data_barcodes[key]['bp'])

features_h0 = np.array(features_h0)
features_bip = np.array(features_bip)
features_h1 = np.array(features_h1)
embeddings = np.array(embeddings)
labels_mf = np.array(labels_mf)
labels_cc = np.array(labels_cc)
labels_bp = np.array(labels_bp)

kf = KFold(n_splits=5, shuffle=True, random_state=92)

mf_f1s = {'embeds': [], 'bip': [], 'h0': [], 'h1': []}
cc_f1s = {'embeds': [], 'bip': [], 'h0': [], 'h1': []}
bp_f1s = {'embeds': [], 'bip': [], 'h0': [], 'h1': []}

base_clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
# base_clf = DecisionTreeClassifier(max_depth=100, max_features='sqrt', random_state=92)

for train_index, test_index in kf.split(features_h0):
    n_samples = len(test_index)
    clf = MultiOutputClassifier(base_clf)

    print('MF')
    labels_train = labels_mf[train_index]
    labels_test = labels_mf[test_index]
    train_len = len(train_index)
    test_len = len(test_index)
    start = time.time()
    clf.fit(embeddings[train_index], labels_train)
    probs = clf.predict_proba(embeddings[test_index])
    mf_f1s['embeds'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted embeddings, took {time.time() - start}')
    start = time.time()
    clf.fit(features_h0[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_h0[test_index].reshape(test_len, -1))
    mf_f1s['h0'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted h0, took {time.time() - start}')
    start = time.time()
    clf.fit(features_h1[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_h1[test_index].reshape(test_len, -1))
    mf_f1s['h1'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted h1, took {time.time() - start}')
    start = time.time()
    clf.fit(features_bip[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_bip[test_index].reshape(test_len, -1))
    mf_f1s['bip'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted bipartite, took {time.time() - start}')

    print('CC')
    labels_train = labels_cc[train_index]
    labels_test = labels_cc[test_index]
    start = time.time()
    clf.fit(embeddings[train_index], labels_train)
    probs = clf.predict_proba(embeddings[test_index])
    cc_f1s['embeds'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted embeddings, took {time.time() - start}')
    start = time.time()
    clf.fit(features_h0[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_h0[test_index].reshape(test_len, -1))
    cc_f1s['h0'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted h0, took {time.time() - start}')
    start = time.time()
    clf.fit(features_h1[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_h1[test_index].reshape(test_len, -1))
    cc_f1s['h1'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted h1, took {time.time() - start}')
    start = time.time()
    clf.fit(features_bip[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_bip[test_index].reshape(test_len, -1))
    cc_f1s['bip'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted bipartite, took {time.time() - start}')

    print('BP')
    labels_train = labels_bp[train_index]
    labels_test = labels_bp[test_index]
    start = time.time()
    clf.fit(embeddings[train_index], labels_train)
    probs = clf.predict_proba(embeddings[test_index])
    bp_f1s['embeds'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted embeddings, took {time.time() - start}')
    start = time.time()
    clf.fit(features_h0[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_h0[test_index].reshape(test_len, -1))
    bp_f1s['h0'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted h0, took {time.time() - start}')
    start = time.time()
    clf.fit(features_h1[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_h1[test_index].reshape(test_len, -1))
    bp_f1s['h1'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted h1, took {time.time() - start}')
    start = time.time()
    clf.fit(features_bip[train_index].reshape(train_len, -1), labels_train)
    probs = clf.predict_proba(features_bip[test_index].reshape(test_len, -1))
    bp_f1s['bip'].append(count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in probs]).T, labels_test))
    print(f'\tfitted bipartite, took {time.time() - start}')

print('MF')
for k, v in mf_f1s.items():
    print(f'\tF1-max for {k}: {np.mean(v):.4f}, std: {np.std(v):.4f}')
print('CC')
for k, v in cc_f1s.items():
    print(f'\tF1-max for {k}: {np.mean(v):.4f}, std: {np.std(v):.4f}')
print('BP')
for k, v in bp_f1s.items():
    print(f'\tF1-max for {k}: {np.mean(v):.4f}, std: {np.std(v):.4f}')