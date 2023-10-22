import os
import time
import argparse
import pprint
import pickle
import logging
import json
import copy
import itertools

import pandas as pd
import numpy as np

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import CalibrationDisplay

from calibrator import MyCalibratedClassifier
from common import get_n_jobs
from subgroup_testing import load_test_data
from detector import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="plot calibration ML curves"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help="which features to plot calibration curve for",
    )
    parser.add_argument(
        "--feature-subset-regex",
        type=str,
        default="",
        help="which features to use for subgroup",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.025,
        help="tolerance for model calibration",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=100000,
        help="max number of observations for testing",
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
        "--plot-file-template",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    args.features = args.features.split(",")
    args.dataset = args.dataset_template.replace("SEED",
            str(args.seed))
    args.log_file = args.log_file_template.replace("SEED",
            str(args.seed))
    args.mdl_file = args.mdl_file_template.replace("SEED",
            str(args.seed))
    args.plot_file = (args.plot_file_template.replace("SEED",
            str(args.seed)) if args.plot_file_template is not None else None)
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

    np_X, np_Y, true_mu, feature_names, test_axes = load_test_data(args)
    np_X = np_X[:,ml_mdl.train_axes]
    
    # Hack for zsfg, but I guess it works for other code too
    feature_names = np.array([f_name.replace("simpleimputer__demographic_", "").replace("pipeline-1__demographic_", "").replace("remainder__", "") for f_name in feature_names])
    print(feature_names)
    
    
    sns.set_context('paper', font_scale=1.8)
    BINS = 3
    if args.features:
        _, axs = plt.subplots(nrows=len(args.features),
                ncols=BINS,figsize=(BINS * 5, len(args.features) * 3.5))
        for feat_idx, feat in enumerate(args.features):
            print("FEATURE", feat)
            feat_ax = np.where(feature_names == feat)[0][0]
            if np.unique(np_X[:,feat_ax]).size == 2:
                feat_bins = [0,0.5]
            else:
                feat_bins = np.quantile(np_X[:,feat_ax], q=np.arange(0,1,0.99/BINS))
            print("feat_bins", feat_bins)

            # plt.clf()
            # plt.hist(np_X[:, feat_ax])
            # plt.show()

            for bin_idx in range(len(feat_bins) - 1):
                display_name = "%s %.1f-%.1f" % (
                    feat.capitalize(),
                    feat_bins[bin_idx],
                    feat_bins[bin_idx + 1])
                feat_mask = (np_X[:,feat_ax] >= feat_bins[bin_idx]) & (np_X[:,feat_ax] <= feat_bins[bin_idx + 1])
                print(feat_mask.sum())
                curr_ax = axs[feat_idx, bin_idx] if len(args.features) > 1 else axs[bin_idx]
                res = CalibrationDisplay.from_estimator(
                    ml_mdl,
                    np_X[feat_mask],
                    np_Y.flatten()[feat_mask],
                    n_bins=7,
                    strategy='quantile',
                    name=display_name,
                    ax=curr_ax,
                    ref_line=False)
                # ref line
                curr_ax.plot([0, 1], [-args.tolerance, 1 - args.tolerance], "k:", label="ref")
                curr_ax.plot([0, 1], [args.tolerance, 1 + args.tolerance], "k:", label="ref")
                
                bin_edges = [0] + [(res.prob_pred[i] + res.prob_pred[i + 1])/2 for i in range(res.prob_pred.size - 1)] + [1]
                preds = ml_mdl.predict_proba(np_X[feat_mask])[:,1]
                bin_sizes = [((preds >= bin_edges[i]) & (preds < bin_edges[i + 1])).sum() for i in range(len(bin_edges) - 1)]
                print("BIN SIZE", bin_sizes)
                std_errs = np.sqrt(res.prob_true * (1 - res.prob_true)/bin_sizes)
                curr_ax.fill_between(res.prob_pred, res.prob_true-1.96 * std_errs, res.prob_true+1.96 * std_errs, alpha=0.3)
                curr_ax.set_xlabel('Mean predicted probability')
                curr_ax.set_ylabel('Event rate')
                curr_ax.set_xlim((0,1))
                curr_ax.set_ylim((0,1))
                curr_ax.set_title("%s [%.1f,%.1f]" % (feat, feat_bins[bin_idx], feat_bins[bin_idx + 1]))
                curr_ax.get_legend().remove()
            sns.despine()
        plt.tight_layout()
    else:
        CalibrationDisplay.from_estimator(
            ml_mdl,
            np_X,
            np_Y.flatten(),
            n_bins=10,
            name="orig_pred")
    plt.savefig(args.plot_file)
    print(args.plot_file)

if __name__ == "__main__":
    main()
