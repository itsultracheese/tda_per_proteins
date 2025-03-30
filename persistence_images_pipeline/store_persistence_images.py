import os

import numpy as np
import pandas as pd
from gudhi.representations import DiagramSelector, PersistenceImage


def read_pickle_file(file):
    return pd.read_pickle(file)


def extract_data(folder, max_files=-1):
    barcodes, labels = [], []
    count = 0

    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if count >= max_files and max_files != -1:
                break

            file_path = os.path.join(subdir, file)
            cur_file = read_pickle_file(file_path)

            prot = list(cur_file.keys())[0]
            barcode = cur_file[prot]["barcodes"]

            mx_len = max([len(b) for b in barcode])
            for i in range(len(barcode)):
                if len(barcode[i]) < mx_len:
                    barcode[i] += [[0, 0, 0] for _ in range(mx_len - len(barcode[i]))]
            barcode = np.array(barcode)

            barcodes.append(barcode)
            lab = {
                "MF": cur_file[prot]["label_MF"],
                "CC": cur_file[prot]["label_CC"],
                "BP": cur_file[prot]["label_BP"],
            }
            labels.append(lab)
            count += 1

    return barcodes, labels


def compute_persistence_image(barcode):
    pds = DiagramSelector(use=True).fit_transform(barcode)
    # precomputed bounds
    im_bnds = [0.0, 0.05, 0.8, 1.0]
    PI_params = {
        "bandwidth": 8e-3,
        "weight": lambda x: x[1],
        "resolution": [50, 50],
        "im_range": im_bnds,
    }
    return PersistenceImage(**PI_params).fit_transform(pds)


def main():
    folder = "/barcodes_valid_corrected"
    barcodes, labels = extract_data(folder)

    image_folder = "/valid"
    label_folder = "./validation"

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    for i, (barcode, label) in enumerate(zip(barcodes, labels)):
        persistence_image = compute_persistence_image(barcode)

        flat_img = persistence_image.flatten()
        image_path = os.path.join(image_folder, f"persistence_image_{i}.npy")
        np.save(image_path, flat_img)

        label_path = os.path.join(label_folder, f"{i}.npz")
        np.savez(label_path, MF=label["MF"], CC=label["CC"], BP=label["BP"])


if __name__ == "__main__":
    main()
