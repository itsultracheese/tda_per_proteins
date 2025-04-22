import os

import numpy as np
import torch
from py_boost import SketchBoost
from tqdm import tqdm


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_protein_vectors(data_dir):
    data = {}
    for file in tqdm(os.listdir(data_dir), desc="Loading protein vectors"):
        if not (file.endswith(".npy") or file.endswith(".npz")):
            continue

        file_id = os.path.splitext(file)[0]
        path = os.path.join(data_dir, file)

        try:
            arr = np.load(path)
            vector = arr["images"] if "images" in arr else arr
        except Exception:
            vector = np.load(path)

        # taking the last N layers
        n_layers = 6
        ids = list(range(len(vector) - 20 * n_layers, len(vector)))
        sub = vector[ids]
        data[file_id] = sub.flatten()

    return data


def load_labels(labels_dir):
    labels = {}
    for file in tqdm(os.listdir(labels_dir), desc="Loading labels"):
        if not file.endswith(".npz"):
            continue
        file_id = os.path.splitext(file)[0]
        path = os.path.join(labels_dir, file)
        try:
            arr = np.load(path)
            labels[file_id] = {
                "MF": arr["MF"],
                "BP": arr["BP"],
                "CC": arr["CC"],
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return labels


def get_common_ids(protein_data, label_data):
    return list(set(protein_data) & set(label_data))


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
    return torch.nan_to_num(all_f1, nan=0.0).max().item()


def train_and_evaluate_pyboost(
    X_train, y_train, X_test, y_test, label_name: str, lr: float
):
    print(f"\nTraining {label_name} @ lr={lr}")
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
        max_bin=128,#256,
        max_depth=10,
        sketch_method='topk',
    ).fit(X_train, y_train, eval_sets=[{"X": X_test, "y": y_test}])
    preds = model.predict(X_test)
    f1 = count_f1_max(preds, y_test)
    print(f"  {label_name} F1 = {f1:.4f}")
    return model, f1


def main():
    set_seeds(42)

    train_protein_dir = "/images"
    train_label_dir = "./train_labels"
    test_protein_dir = "/test_images"
    test_label_dir = "./test_labels"

    prot_train = load_protein_vectors(train_protein_dir)
    lab_train = load_labels(train_label_dir)
    prot_test = load_protein_vectors(test_protein_dir)
    lab_test = load_labels(test_label_dir)

    train_ids = get_common_ids(prot_train, lab_train)
    test_ids = get_common_ids(prot_test, lab_test)

    X_train = np.array([prot_train[i] for i in train_ids])
    y_MF_tr = np.array([lab_train[i]["MF"] for i in train_ids])
    y_BP_tr = np.array([lab_train[i]["BP"] for i in train_ids])
    y_CC_tr = np.array([lab_train[i]["CC"] for i in train_ids])

    X_test = np.array([prot_test[i] for i in test_ids])
    y_MF_te = np.array([lab_test[i]["MF"] for i in test_ids])
    y_BP_te = np.array([lab_test[i]["BP"] for i in test_ids])
    y_CC_te = np.array([lab_test[i]["CC"] for i in test_ids])

    # lrs = [0.1, 0.05, 0.5]
    lrs = [0.1]

    results_MF = {lr: [] for lr in lrs}
    results_BP = {lr: [] for lr in lrs}
    results_CC = {lr: [] for lr in lrs}

    for lr in lrs:
        _, f1_MF = train_and_evaluate_pyboost(
            X_train, y_MF_tr, X_test, y_MF_te, "MF", lr
        )
        _, f1_BP = train_and_evaluate_pyboost(
            X_train, y_BP_tr, X_test, y_BP_te, "BP", lr
        )
        _, f1_CC = train_and_evaluate_pyboost(
            X_train, y_CC_tr, X_test, y_CC_te, "CC", lr
        )
        results_MF[lr].append(f1_MF)
        results_BP[lr].append(f1_BP)
        results_CC[lr].append(f1_CC)

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
