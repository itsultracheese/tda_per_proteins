import os
import time
import pickle
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from scripts.helper_scripts.f1_max_score import count_f1_max

train_folder = ''
test_folder = ''

base_classifier = MultiOutputClassifier(SVC(kernel='linear', random_state=92))
# base_classifier = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3, weights='distance')).fit(train_features, train_labels)

train_files = [x for x in os.listdir(train_folder)]
test_files = [x for x in os.listdir(test_folder)]

train_features = []
train_labels_mf = []
train_labels_cc = []
train_labels_bp = []
for train_file in train_files:
    try:
        with open(os.path.join(train_folder, train_file), 'rb') as f:
            data = pickle.load(f)
            # combine features by layers
            bc = data['betticurves'].reshape((33, 20, -1))
            train_features.append(bc.sum(axis=1).flatten())
            train_labels_mf.append(data['label_MF'])
            train_labels_cc.append(data['label_CC'])
            train_labels_bp.append(data['label_BP'])
    except Exception as e:
        print(e)
        print('File:', train_file)
        print()

train_labels_mf = np.array(train_labels_mf)
train_labels_cc = np.array(train_labels_cc)
train_labels_bp = np.array(train_labels_bp)

test_features = []
test_labels_mf = []
test_labels_cc = []
test_labels_bp = []
for test_file in test_files:
    with open(os.path.join(test_folder, test_file), 'rb') as f:
        data = pickle.load(f)
        bc = data['betticurves'].reshape((33, 20, -1))
        test_features.append(bc.sum(axis=1).flatten())
        test_labels_mf.append(data['label_MF'])
        test_labels_cc.append(data['label_CC'])
        test_labels_bp.append(data['label_BP'])
test_labels_mf = np.array(test_labels_mf)
test_labels_cc = np.array(test_labels_cc)
test_labels_bp = np.array(test_labels_bp)

print('Start fitting MF')
start = time.time()
model_mf = base_classifier.fit(train_features, train_labels_mf)
preds_mf = model_mf.predict_proba(test_features)
n_samples = len(test_features)
f1_score_mf = count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in preds_mf]).T, test_labels_mf)
print('MF:', f1_score_mf, 'took', time.time() - start)

print('Start fitting CC')
start = time.time()
model_cc = base_classifier.fit(train_features, train_labels_cc)
preds_cc = model_cc.predict_proba(test_features)
f1_score_cc = count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in preds_cc]).T, test_labels_cc)
print('CC:', f1_score_cc, 'took', time.time() - start)

print('Start fitting BP')
model_bp = base_classifier.fit(train_features, train_labels_bp)
preds_bp = model_bp.predict_proba(test_features)
f1_score_bp = count_f1_max(np.array([x[:, 1] if x.shape[1] == 2 else np.zeros(n_samples) for x in preds_bp]).T, test_labels_bp)
print('BP:', f1_score_bp, 'took', time.time() - start)
