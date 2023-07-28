import os
import sys
import argparse
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from dataset import create_class_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make dataset for training ML models and auditing ML models"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--orig-beta",
        type=str,
        default=".5",
    )
    parser.add_argument(
        "--orig-intercept",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--new-beta",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--new-intercept",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--num-p",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--n-bigtest",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="simple",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="simple",
    )
    parser.add_argument("--out-file-template", type=str,
            default="_output/dataSEED.pkl")
    args = parser.parse_args()
    args.orig_beta = list(map(float, args.orig_beta.split(",")))
    args.new_beta = list(map(float, args.new_beta.split(","))) if args.new_beta is not None else args.orig_beta
    args.out_file = args.out_file_template.replace("SEED",
            str(args.seed))
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed)

    orig_beta = np.array(args.orig_beta + [0] * (args.num_p - len(args.orig_beta))).reshape((-1, 1))
    new_beta = np.array(args.new_beta + [0] * (args.num_p - len(args.new_beta))).reshape((-1, 1))
    
    # Generate training data
    train_X, train_mu, train_y = create_class_dataset(
        args.train_dataset, args.n_train, orig_beta=orig_beta, new_beta=new_beta, orig_intercept=args.orig_intercept, new_intercept=args.new_intercept
    )
    # Generate target data
    test_X, test_mu, test_y = create_class_dataset(
        args.test_dataset, args.n_test, orig_beta=orig_beta, new_beta=new_beta, orig_intercept=args.orig_intercept, new_intercept=args.new_intercept
    )
    # Generate target data
    new_beta = np.array(args.new_beta + [0] * (args.num_p - len(args.new_beta))).reshape((-1, 1))
    
    with open(args.out_file, "wb") as f:
        pickle.dump({
            "feature_names": [f"x{i}" for i in range(args.num_p)],
            "train": {
                "X": train_X,
                "mu": train_mu,
                "y": train_y,
            },
            "test": {
                "X": test_X,
                "mu": test_mu,
                "y": test_y,
            },
            "bigtest": {
                "X": test_X,
                "mu": test_mu,
                "y": test_y,
            },
        }, f)

if __name__ == "__main__":
    main()
