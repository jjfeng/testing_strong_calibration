import os
import re
import time
import argparse
import pprint
import pickle
import logging
import copy
import itertools
import json

import pandas as pd
import numpy as np

import seaborn as sns

from detector import *
from subgroup_testing import load_test_data, create_detector


def parse_args():
    parser = argparse.ArgumentParser(description="calculate feature importance, store in CSV")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dataset-template",
        type=str,
        default="_output/dataSEED.pkl",
    )
    parser.add_argument(
        "--mdl-file-template",
        type=str,
        default="_output/mdlSEED.pkl",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logSEED.txt",
    )
    parser.add_argument(
        "--detector-file-template",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--res-file-template",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--feature-file-template",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    args.dataset = args.dataset_template.replace("SEED", str(args.seed))
    args.mdl_file = args.mdl_file_template.replace("SEED", str(args.seed))
    args.detector_file = args.detector_file_template.replace("SEED", str(args.seed))
    args.res_file = args.res_file_template.replace("SEED", str(args.seed))
    args.log_file = args.log_file_template.replace("SEED", str(args.seed))
    args.feature_subset_regex = None
    args.feature_file = (
        args.feature_file_template.replace("SEED", str(args.seed))
        if args.feature_file_template is not None
        else None)
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    # Load the ML model to evaluate
    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)

    # Load test data
    np_X, np_Y, true_mu, feature_names, test_axes = load_test_data(args)

    # load detector
    with open(args.detector_file, "rb") as f:
        detector = pickle.load(f)
    
    test_stats_df = pd.read_csv(args.res_file)

    # get feature importance
    feat_imports_df = detector.get_feat_imports(test_stats_df, ml_mdl, np_X, np_Y, n_repeats=args.n_repeats)
    if feat_imports_df is not None:
        feat_imports_df.to_csv(args.feature_file, index=False)

if __name__ == "__main__":
    main()
