import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from py_boost import SketchBoost


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_protein_vectors(data_dir):
    data = {}
    files = os.listdir(data_dir)
    for file in tqdm(files, desc="Loading protein vectors"):
        if file.endswith(".npy") or file.endswith(".npz"):
            file_id = os.path.splitext(file)[0]
            file_path = os.path.join(data_dir, file)
            try:
                vector = np.load(file_path)["images"]
            except:
                vector = np.load(file_path)
            
            # taking only the last 20 entries (e.g., last attention layers)
            n_layers = 1
            vector = np.array(
                [vector[i] for i in range(len(vector) - 20 * n_layers, len(vector))]
            )
            vector = vector.flatten()
            data[file_id] = vector
    return data


def load_labels(labels_dir):
    labels = {}
    files = os.listdir(labels_dir)
    for file in tqdm(files, desc="Loading labels"):
        if file.endswith(".npz"):
            file_id = os.path.splitext(file)[0]
            file_path = os.path.join(labels_dir, file)
            try:
                label_data = np.load(file_path)
                labels[file_id] = {
                    "MF": label_data["MF"],
                    "BP": label_data["BP"],
                    "CC": label_data["CC"],
                }
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return labels


def get_common_ids(protein_data, label_data):
    protein_ids = set([x.split("_")[2] for x in protein_data.keys()])
    label_ids = set(label_data.keys())
    return list(protein_ids.intersection(label_ids))


def count_f1_max(pred, target) -> float:
    pred = torch.Tensor(pred)
    target = torch.Tensor(target)
    if target.sum() == 0:
        return 0.0
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = True
    is_start = torch.scatter(is_start, 1, order, is_start)
    all_order = pred.flatten().argsort(descending=True)
    order = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    ).flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    all_f1 = torch.nan_to_num(all_f1, nan=0.0)
    return all_f1.max().item()


def train_and_evaluate_pyboost(
    X_train, y_train, X_test, y_test, label_name: str, lr: float
):
    print(f"Training for {label_name} with learning rate {lr}")
    model = SketchBoost(
        loss="multilabel",
        metric="f1",
        ntrees=10000,
        lr=lr,
        verbose=300,
        es=300,
        lambda_l2=1,
        gd_steps=1,
        subsample=1,
        colsample=1,
        min_data_in_leaf=10,
        max_bin=256,
        max_depth=10,
    ).fit(X_train, y_train, eval_sets=[{"X": X_test, "y": y_test}])

    preds = model.predict(X_test)
    f1 = count_f1_max(preds, y_test)
    print(f"F1 for {label_name} using PyBoost with lr={lr}: {f1:.4f}\n")
    return model, f1


def main():
    set_seeds(42)

    protein_dir = "/valid"
    label_dir = "./validation"

    protein_data = load_protein_vectors(protein_dir)
    label_data = load_labels(label_dir)
    common_ids = get_common_ids(protein_data, label_data)

    X = np.array([protein_data[f"persistence_image_{_id}"] for _id in common_ids])
    y_MF = np.array([label_data[_id]["MF"] for _id in common_ids])
    y_BP = np.array([label_data[_id]["BP"] for _id in common_ids])
    y_CC = np.array([label_data[_id]["CC"] for _id in common_ids])

    lrs = [0.1, 0.05, 0.5]
    
    results_MF = {lr: [] for lr in lrs}
    results_BP = {lr: [] for lr in lrs}
    results_CC = {lr: [] for lr in lrs}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_idx, test_idx in kf.split(X):
        print(f"\nFold {fold}/{kf.get_n_splits()}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_MF_train, y_MF_test = y_MF[train_idx], y_MF[test_idx]
        y_BP_train, y_BP_test = y_BP[train_idx], y_BP[test_idx]
        y_CC_train, y_CC_test = y_CC[train_idx], y_CC[test_idx]

        for lr in lrs:
            print(f"\n-- Learning Rate: {lr} --")
            _, f1_MF = train_and_evaluate_pyboost(
                X_train, y_MF_train, X_test, y_MF_test, "MF", lr
            )
            _, f1_BP = train_and_evaluate_pyboost(
                X_train, y_BP_train, X_test, y_BP_test, "BP", lr
            )
            _, f1_CC = train_and_evaluate_pyboost(
                X_train, y_CC_train, X_test, y_CC_test, "CC", lr
            )
            results_MF[lr].append(f1_MF)
            results_BP[lr].append(f1_BP)
            results_CC[lr].append(f1_CC)

        fold += 1

    print("\nResults")
    for lr in lrs:
        avg_f1_MF = np.mean(results_MF[lr])
        avg_f1_BP = np.mean(results_BP[lr])
        avg_f1_CC = np.mean(results_CC[lr])

        std_f1_MF = np.std(results_MF[lr])
        std_f1_BP = np.std(results_BP[lr])
        std_f1_CC = np.std(results_CC[lr])

        print(f"Learning Rate {lr}:")
        print(f"  MF: Average F1 = {avg_f1_MF:.4f} (std = {std_f1_MF:.4f})")
        print(f"  BP: Average F1 = {avg_f1_BP:.4f} (std = {std_f1_BP:.4f})")
        print(f"  CC: Average F1 = {avg_f1_CC:.4f} (std = {std_f1_CC:.4f})\n")


if __name__ == "__main__":
    main()
