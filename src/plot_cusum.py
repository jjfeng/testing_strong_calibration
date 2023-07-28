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
from subgroup_testing import load_test_data


def parse_args():
    parser = argparse.ArgumentParser(description="plot cusum plots")
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
        "--plot-input-file-template",
        type=str,
        default="_output/plotSEED.csv",
    )
    parser.add_argument(
        "--plot-output-file-template",
        type=str,
        default="_output/plotSEED.png",
    )
    parser.add_argument(
        "--detector-file-template",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    args.dataset = args.dataset_template.replace("SEED", str(args.seed))
    args.mdl_file = args.mdl_file_template.replace("SEED", str(args.seed))
    args.detector_file = args.detector_file_template.replace("SEED", str(args.seed))
    args.feature_subset_regex = None
    args.plot_input_file = args.plot_input_file_template.replace("SEED", str(args.seed))
    args.plot_output_file = args.plot_output_file_template.replace("SEED", str(args.seed))
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load the ML model to evaluate
    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)

    # Load test data
    np_X, np_Y, true_mu, feature_names, test_axes = load_test_data(args)

    # load detector
    with open(args.detector_file, "rb") as f:
        detector = pickle.load(f)
    
    plot_df = pd.read_csv(args.plot_input_file)

    # plot!
    sns.set_context('paper', font_scale=1.8)
    detector.plot(plot_df, args.plot_output_file)

if __name__ == "__main__":
    main()
