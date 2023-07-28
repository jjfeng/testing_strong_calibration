import os
import re
import time
import argparse
import pprint
import pickle
import logging
import json

import pandas as pd
import numpy as np
from scipy.sparse import isspmatrix

import seaborn as sns

from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from common import *
from detector import *


def parse_args():
    parser = argparse.ArgumentParser(description="Test for strong calibration")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed number for this test (will look for matching seed for data and ML model)",
    )
    parser.add_argument(
        "--detection-models",
        type=str,
        help="comma-separated strings indicating which residual models to fit",
    )
    parser.add_argument(
        "--detection-params",
        type=str,
        default="testing_dict.json",
        help="file name containing the hyperparameters to consider for each residual model",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Type I error")
    parser.add_argument(
        "--detector",
        type=str,
        default="CVScore",
        help="Which test to run",
        choices=[
            "SplitScore",
            "SplitScoreTwoSided",
            "CVScore",
            "CVScoreTwoSided",
            "SplitChiSquared",
            "SplitChiSquaredTwoSided",
            "CVChiSquared",
            "CVChiSquaredTwoSided",
            "HosmerLemeshow",
            "LogisticRecalibration",
            "LogisticRecalibrationTwoSided",
        ],
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=500,
        help="Maximum number of audit observations to read in from the dataset",
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="If running cv, the number of folds"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.75,
        help="If sample-splitting, proportion of data for training the residual models",
    )
    parser.add_argument(
        "--tolerance-prob",
        type=float,
        default=0,
        help="tolerance for predicted probabilities",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=200,
        help="number of bootstrap samples for calculating p-values",
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default="greater",
        choices=["greater", "less", "both"],
    )
    parser.add_argument(
        "--feature-subset-regex",
        type=str,
        default=None,
        help="regex for indicating which subset of features to use for testing strong calibration",
    )
    parser.add_argument(
        "--dataset-template",
        type=str,
        default="_output/dataSEED.pkl",
        help="pickle file containing the dataset (if SEED is in the file name, we replace SEED with the provided seed number)",
    )
    parser.add_argument(
        "--mdl-file-template",
        type=str,
        default="_output/mdlSEED.pkl",
        help="pickle file containing the dataset",
    )
    parser.add_argument(
        "--res-file-template",
        type=str,
        default="_output/resSEED.csv",
        help="output file containing test results",
    )
    parser.add_argument(
        "--log-file-template", type=str, default="_output/logSEED.txt", help="log file"
    )
    parser.add_argument(
        "--detector-file-template",
        type=str,
        default=None,
        help="pickle file to save the test class, if desired",
    )
    parser.add_argument("--plot-file-template", type=str, default=None)
    args = parser.parse_args()
    args.detection_models = args.detection_models.split("_")
    args.axes = [
        -1
    ]  # default pre-specified axis to use for HL and Platt testing procedures
    args.dataset = args.dataset_template.replace("SEED", str(args.seed))
    args.mdl_file = args.mdl_file_template.replace("SEED", str(args.seed))
    args.detector_file = (
        args.detector_file_template.replace("SEED", str(args.seed))
        if args.detector_file_template is not None
        else None
    )
    args.plot_file = (
        args.plot_file_template.replace("SEED", str(args.seed))
        if args.plot_file_template is not None
        else None
    )
    args.log_file = args.log_file_template.replace("SEED", str(args.seed))
    args.res_file = args.res_file_template.replace("SEED", str(args.seed))
    return args


def load_test_data(args):
    # Load the test data
    with open(args.dataset, "rb") as f:
        data_dict = pickle.load(f)
        feature_names = data_dict["feature_names"]

    if isspmatrix(data_dict["test"]["X"]):
        data_dict["test"]["X"] = data_dict["test"]["X"].toarray()

    assert data_dict["test"]["X"].shape[0] >= args.n_test
    np_X = data_dict["test"]["X"][: args.n_test]
    np_Y = data_dict["test"]["y"].flatten()[: args.n_test]
    logging.info("dataset size %s", np_X.shape)

    shuffle_idx = np.random.choice(np_X.shape[0], np_X.shape[0], replace=False)
    true_mu_shuffled = (
        data_dict["test"]["mu"].flatten()[shuffle_idx]
        if "mu" in data_dict["test"]
        else None
    )

    test_axes = np.ones(np_X.shape[1], dtype=bool)
    if args.feature_subset_regex is not None:
        test_axes = np.array(
            [
                re.fullmatch(args.feature_subset_regex, w) is not None
                for w in feature_names
            ]
        )

    return (
        np_X[shuffle_idx],
        np_Y[shuffle_idx],
        true_mu_shuffled,
        np.array(feature_names),
        test_axes,
    )


def print_diagnostics(ml_mdl, np_X, true_mu, tolerance_prob):
    """
    Random testing just for sanity checks
    """
    pred_prob = ml_mdl.predict_proba(np_X)[:, 1]
    null_prob = make_prob(pred_prob + tolerance_prob)
    true_grouping = true_mu > null_prob
    residuals = true_mu - pred_prob
    logging.info("greater group size %f", true_grouping.mean())
    if true_grouping.sum() > 3:
        logging.info(
            "greater residuals %s",
            np.quantile(residuals[true_grouping], q=[0.5, 0.75, 0.9]),
        )
    null_prob = make_prob(pred_prob - tolerance_prob)
    true_grouping = null_prob > true_mu
    logging.info("less group size %f", true_grouping.mean())
    if true_grouping.sum() > 3:
        logging.info(
            "less residuals %s",
            np.quantile(-residuals[true_grouping], q=[0.5, 0.75, 0.9]),
        )


def create_detector(feature_names, test_axes, args):
    """
    Instatiate the class for performing the test
    """
    n_jobs = get_n_jobs()
    print("NJOBS", n_jobs)

    detection_mdl_dict = {}
    if "RandomForestRegressor" in args.detection_models:
        detection_mdl_dict["RandomForestRegressor"] = RandomForestRegressor(
            n_jobs=n_jobs, random_state=0
        )
    if "KernelLogistic" in args.detection_models:
        clf = Pipeline(
            [
                ("nys", Nystroem(kernel="polynomial")),
                ("lr", LogisticRegression(solver="liblinear")),
            ]
        )
        detection_mdl_dict["KernelLogistic"] = clf

    detect_param_dict = None
    if args.detection_params:
        with open(args.detection_params, "r") as f:
            detect_param_dict = json.load(f)

    if args.detector == "LogisticRecalibration":
        detector = LogisticRecalibration(
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "LogisticRecalibrationTwoSided":
        detector = LogisticRecalibrationTwoSided(
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "SplitScore":
        detector = SplitScore(
            detection_mdl_dict,
            feature_names,
            test_axes,
            detect_param_dict,
            args.split,
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "SplitScoreTwoSided":
        detector = SplitScoreTwoSided(
            detection_mdl_dict,
            feature_names,
            test_axes,
            detect_param_dict,
            args.split,
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "CVScore":
        detector = CVScore(
            detection_mdl_dict,
            feature_names,
            test_axes,
            detect_param_dict,
            args.cv,
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "CVScoreTwoSided":
        detector = CVScoreTwoSided(
            detection_mdl_dict,
            feature_names,
            test_axes,
            detect_param_dict,
            args.cv,
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
        # print(CVScoreTwoSided.__mro__)
    elif args.detector == "SplitChiSquared":
        detector = SplitChiSquared(
            detection_mdl_dict,
            feature_names,
            test_axes,
            detect_param_dict,
            args.split,
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "CVChiSquared":
        detector = CVChiSquared(
            detection_mdl_dict,
            feature_names,
            test_axes,
            detect_param_dict,
            args.cv,
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "SplitChiSquaredTwoSided":
        detector = SplitChiSquaredTwoSided(
            detection_mdl_dict,
            feature_names,
            test_axes,
            detect_param_dict,
            args.split,
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "CVChiSquaredTwoSided":
        detector = CVChiSquaredTwoSided(
            detection_mdl_dict,
            feature_names,
            test_axes,
            detect_param_dict,
            args.cv,
            args.n_boot,
            args.tolerance_prob,
            alternative=args.alternative,
        )
    elif args.detector == "HosmerLemeshow":
        detector = HosmerLemeshow(
            args.axes,
            args.n_boot,
            tolerance_prob=args.tolerance_prob,
            alternative=args.alternative,
        )

    return detector


def main():
    args = parse_args()
    np.random.seed(args.seed)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    sns.set_context(context="paper", font_scale=2)
    st_time = time.time()

    # Load the ML model to evaluate
    with open(args.mdl_file, "rb") as f:
        ml_mdl = pickle.load(f)

    # Load test data
    np_X, np_Y, true_mu, feature_names, test_axes = load_test_data(args)

    # Print some diagnostics
    if true_mu is not None:
        print_diagnostics(ml_mdl, np_X, true_mu, args.tolerance_prob)

    # Initialize the detector
    detector = create_detector(feature_names, test_axes, args)
    # HACK: DEBUGGING ONLY
    # detector.true_mu = true_mu

    # Run the test
    res, plot_df = detector.test(
        np_X,
        np_Y,
        ml_mdl,
    )
    logging.info("run time %d", int(time.time() - st_time))

    print(res)
    logging.info(res[["method", "pval"]])
    res.to_csv(args.res_file, index=False)

    # Save results
    if args.plot_file is not None:
        plot_df.to_csv(args.plot_file, index=False)

    if args.detector_file:
        detector.clean_for_saving(res)
        with open(args.detector_file, "wb") as f:
            pickle.dump(detector, f)


if __name__ == "__main__":
    main()
