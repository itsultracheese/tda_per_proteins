import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

from helper_scripts.f1_max_score import count_f1_max

train_folder = './src/landscapes/train_for_models'
test_folder = './src/landscapes/test_for_models'

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
            train_features.append(data['persistence_landscapes'].flatten())
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
        test_features.append(data['persistence_landscapes'].flatten())
        test_labels_mf.append(data['label_MF'])
        test_labels_cc.append(data['label_CC'])
        test_labels_bp.append(data['label_BP'])
test_labels_mf = np.array(test_labels_mf)
test_labels_cc = np.array(test_labels_cc)
test_labels_bp = np.array(test_labels_bp)


n_neighbors_list = [3, 5, 7]  
weights_list = ['uniform', 'distance'] 
leaf_size_list = [10, 30, 50] 

reports = {'MF': '', 'CC': '', 'BP': ''}


print('Start fitting MF')
for n_neighbors in n_neighbors_list:
    for weights in weights_list:
        for leaf_size in leaf_size_list:
            
            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                leaf_size=leaf_size,
                algorithm='auto',
                metric='minkowski'
            )
            

            clf_mf = MultiOutputClassifier(knn).fit(train_features, train_labels_mf)

            preds_mf = clf_mf.predict_proba(np.array(test_features))

            f1_score_mf = count_f1_max(preds_mf, test_labels_mf)

            reports['MF'] += (
                f"MF: {f1_score_mf:.4f}, "
                f"n_neighbors: {n_neighbors}, "
                f"weights: {weights}, "
                f"leaf_size: {leaf_size}\n"
            )


print('Start fitting CC')
for n_neighbors in n_neighbors_list:
    for weights in weights_list:
        for leaf_size in leaf_size_list:
            
            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                leaf_size=leaf_size,
                algorithm='auto',
                metric='minkowski'
            )
            

            clf_cc = MultiOutputClassifier(knn).fit(train_features, train_labels_cc)

            preds_cc = clf_cc.predict_proba(np.array(test_features))

            f1_score_cc = count_f1_max(preds_cc, test_labels_cc)

            reports['CC'] += (
                f"CC: {f1_score_cc:.4f}, "
                f"n_neighbors: {n_neighbors}, "
                f"weights: {weights}, "
                f"leaf_size: {leaf_size}\n"
            )

print('Start fitting BP')
for n_neighbors in n_neighbors_list:
    for weights in weights_list:
        for leaf_size in leaf_size_list:
            
            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                leaf_size=leaf_size,
                algorithm='auto',
                metric='minkowski'
            )
            

            clf_bp = MultiOutputClassifier(knn).fit(train_features, train_labels_bp)

            preds_bp = clf_bp.predict_proba(np.array(test_features))

            f1_score_bp = count_f1_max(preds_bp, test_labels_bp)

            reports['MF'] += (
                f"MF: {f1_score_bp:.4f}, "
                f"n_neighbors: {n_neighbors}, "
                f"weights: {weights}, "
                f"leaf_size: {leaf_size}\n"
            )

for key in reports:
    print(reports[key])
    print()