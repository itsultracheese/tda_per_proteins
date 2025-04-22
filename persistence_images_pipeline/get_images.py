import os
import pickle

import numpy as np
from gudhi.representations import DiagramSelector, PersistenceImage
from tqdm import tqdm


def get_per_im(barcodes):
    pds = DiagramSelector(use=True).fit_transform(barcodes)
    vpdtr = np.vstack(pds)

    pers = vpdtr[:, 1] - vpdtr[:, 0]
    im_bnds = [np.min(vpdtr[:, 0]), 0.05, 0.8, np.max(pers)]
    PI_params = {
        "bandwidth": 2e-2,
        "weight": lambda x: x[1],
        "resolution": [25, 25],
        "im_range": im_bnds,
    }
    image = PersistenceImage(**PI_params).fit_transform(pds)
    return image


def read_one_file(file_path, barcodes=True, labels=False):
    with open(file_path, "rb") as file:
        seq_data = pickle.load(file)

    for sequence, data in seq_data.items():
        if barcodes:
            barcodes = data["barcodes"]
        if labels:
            label_MF = data["label_MF"]
            label_BP = data["label_BP"]
            label_CC = data["label_CC"]

    ans = {}

    if barcodes:
        try:
            barcodes = np.array(barcodes)
        except ValueError:
            for idx, el in enumerate(barcodes):
                cur = len(el)
                prev = None

                if idx > 0:
                    diff = cur - prev
                    if diff < 0:
                        for _ in range(-diff):
                            barcodes[idx].append([0, 0, 0])
                    elif diff > 0:
                        for _ in range(diff):
                            barcodes[idx].append([0, 0, 0])

                prev = len(el)

            barcodes = np.array(barcodes)

        PI = get_per_im(barcodes)
        ans["images"] = PI

    if labels:
        label_MF = np.array(label_MF)
        label_CC = np.array(label_CC)
        label_BP = np.array(label_BP)
        ans["MF"] = label_MF
        ans["CC"] = label_CC
        ans["BP"] = label_BP

    return ans


def read_all_files(dir_path, output_dir, barcodes=True, labels=False, verbose=True):
    files = sorted(
        [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))],
        key=lambda x: int(x[:-4]),
    )
    total_files = len(files)

    for filename in tqdm(
        files, total=total_files, desc="Processing files", disable=not verbose
    ):
        file_path = os.path.join(dir_path, filename)
        result = read_one_file(file_path, barcodes=barcodes, labels=labels)

        base_name = os.path.splitext(filename)[0]
        output_file_path = os.path.join(output_dir, f"{base_name}.npz")
        np.savez(output_file_path, **result)


if __name__ == "__main__":
    dir_path = "/processed_test/barcodes"
    output_dir = "/test_images"

    os.makedirs(output_dir, exist_ok=True)

    read_all_files(
        dir_path=dir_path,
        output_dir=output_dir,
        barcodes=True,
        labels=False,
    )
