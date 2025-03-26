from py_boost import SketchBoost
import os
import pickle
import numpy as np

from scripts.helper_scripts.f1_max_score import count_f1_max

train_folder = ''
test_folder = ''

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
            train_features.append(data['betticurves'].flatten())
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
        test_features.append(data['betticurves'].flatten())
        test_labels_mf.append(data['label_MF'])
        test_labels_cc.append(data['label_CC'])
        test_labels_bp.append(data['label_BP'])
test_labels_mf = np.array(test_labels_mf)
test_labels_cc = np.array(test_labels_cc)
test_labels_bp = np.array(test_labels_bp)


lrs = [0.1, 0.05, 0.01]
ntrees = [10000, 20000]
maxdepths = [6, 8, 10]

reports = {'MF': '', 'CC': '', 'BP': ''}

print('Start fitting MF')
for lr in lrs:
    for ntree in ntrees:
        for max_depth in maxdepths:
            model_mf = SketchBoost(
                    loss='multilabel', metric='f1',
                    ntrees=ntree, lr=lr, verbose=300, es=300, lambda_l2=1, gd_steps=1, 
                    subsample=1, colsample=1, min_data_in_leaf=10, 
                    max_bin=256, max_depth=max_depth,
                ).fit(train_features, train_labels_mf, eval_sets = [{'X': np.array(test_features), 'y': test_labels_mf}])
            preds_mf = model_mf.predict(np.array(test_features))
            f1_score_mf = count_f1_max(np.array(preds_mf), test_labels_mf)
            reports['MF'] += 'MF:' + str(f1_score_mf) + 'lr:' + str(lr) + ' ntree:' + str(ntree) + ' max_depth:' + str(max_depth) + '\n'

print('Start fitting CC')
for lr in lrs:
    for ntree in ntrees:
        for max_depth in maxdepths:
            model_cc = SketchBoost(
                    loss='multilabel', metric='f1',
                    ntrees=ntree, lr=lr, verbose=300, es=300, lambda_l2=1, gd_steps=1, 
                    subsample=1, colsample=1, min_data_in_leaf=10, 
                    max_bin=256, max_depth=max_depth,
                ).fit(train_features, train_labels_cc, eval_sets = [{'X': np.array(test_features), 'y': test_labels_cc}])
            preds_cc = model_cc.predict(np.array(test_features))
            f1_score_cc = count_f1_max(np.array(preds_cc), test_labels_cc)
            reports['CC'] += 'CC:' + str(f1_score_cc) + 'lr:' + str(lr) + ' ntree:' + str(ntree) + ' max_depth:' + str(max_depth) + '\n'

print('Start fitting BP')
for lr in lrs:
    for ntree in ntrees:
        for max_depth in maxdepths:
            model_bp = SketchBoost(
                    loss='multilabel', metric='f1',
                    ntrees=ntree, lr=lr, verbose=300, es=300, lambda_l2=1, gd_steps=1, 
                    subsample=1, colsample=1, min_data_in_leaf=10, 
                    max_bin=256, max_depth=max_depth,
                ).fit(train_features, train_labels_bp, eval_sets = [{'X': np.array(test_features), 'y': test_labels_bp}])
            preds_bp = model_bp.predict(np.array(test_features))
            f1_score_bp = count_f1_max(np.array(preds_bp), test_labels_bp)
            reports['BP'] += 'BP:' + str(f1_score_bp) + 'lr:' + str(lr) + ' ntree:' + str(ntree) + ' max_depth:' + str(max_depth) + '\n'

for key in reports:
    print(reports[key])
    print()
