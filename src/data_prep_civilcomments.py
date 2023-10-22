import os
import pickle
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Read data from civil comments and make embeddings")
    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.75
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/civil_comments.csv",
    )
    parser.add_argument(
        "--out-file-template",
        type=str,
        default="_output/zsfg_data_SEED.pkl",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logSEED.txt",
    )
    args = parser.parse_args()
    args.out_file = args.out_file_template.replace('SEED', str(args.seed))
    args.log_file = args.log_file_template.replace('SEED', str(args.seed))
    return args

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    np.random.seed(args.seed)

    civil_df = pd.read_csv(
        args.data_file,
        index_col=False
    )
    print(civil_df)
    print("TOTAL data size", civil_df.shape)
    logging.info(f"TOTAL data size {civil_df.shape}")

    logging.info("prevalence %f", civil_df.y.mean())

    shuffled_all_idxs = np.arange(civil_df.shape[0])
    np.random.shuffle(shuffled_all_idxs)
    print("shuffled_all_idxs", shuffled_all_idxs)
    
    if args.max_sentences is not None:
        civil_df = civil_df.iloc[shuffled_all_idxs[:args.max_sentences]].reset_index(drop=True)
        logging.info(f"data processed {civil_df.shape[0]}")

    X = civil_df.iloc[:,:-1]

    # Split train test
    n_train = int(args.train_frac * civil_df.shape[0])
    all_idx = np.arange(civil_df.shape[0])
    train_idx = all_idx[:n_train]
    test_idx = all_idx[n_train:]
    print(civil_df)
    
    with open(args.out_file, "wb") as f:
        pickle.dump({
                "train": {
                    "X": X.iloc[train_idx].to_numpy(),
                    "y": civil_df.y[train_idx].to_numpy(),
                },
                "test": {
                    "X": X.iloc[test_idx].to_numpy(),
                    "y": civil_df.y[test_idx].to_numpy(),
                },
                "feature_names": civil_df.columns[:-1],
            },
            f,
        )

if __name__ == "__main__":
    main()
