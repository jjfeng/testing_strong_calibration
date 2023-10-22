import os
import re
import time
import argparse
import pprint
import pickle
import logging
import json
import copy
import itertools
import scipy
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay

from common import get_n_jobs
from detector import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="train a ML algorithm"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LogisticRegression",
        choices=["RandomForestClassifier", "LogisticRegression", "GradientBoostingClassifier", "MLPClassifier"]
    )
    parser.add_argument("--param-dict-file", type=str, default="model_dict.json")
    parser.add_argument(
        "--calib-cv",
        type=int,
        default=5,
        help="CV folds for calib",
    )
    parser.add_argument(
        "--calib-method",
        type=str,
        choices=["sigmoid", "isotonic"],
        default="sigmoid"
    )
    parser.add_argument(
        "--feature-subset-regex",
        type=str,
        default=None,
        help="regex for indicating which subset of features to use for fitting model",
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
    args.dataset_file = args.dataset_template.replace("SEED",
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

    # Generate training data
    with open(args.dataset_file, "rb") as f:
        data_dict = pickle.load(f)
        logging.info("prevalence %f", data_dict["train"]["y"].mean())
        feature_names = data_dict['feature_names']
        if args.feature_subset_regex is not None:
            train_axes = np.array(
                [
                    re.fullmatch(args.feature_subset_regex, w) is not None
                    for w in feature_names
                ]
            )
        else:
            train_axes = np.arange(data_dict["train"]["X"].shape[1])

    with open(args.param_dict_file, "r") as f:
        full_param_dict = json.load(f)
        param_dict = full_param_dict[args.model_type]

    # Train the original ML model
    n_jobs = get_n_jobs()
    print("n_jobs", n_jobs)
    if args.model_type == "GradientBoostingClassifier":
        base_mdl = GradientBoostingClassifier()
    elif args.model_type == "RandomForestClassifier":
        base_mdl = RandomForestClassifier(n_jobs=n_jobs)
    elif args.model_type == "LogisticRegression":
        base_mdl = LogisticRegression(penalty="l1", solver="saga")
    elif args.model_type == "MLPClassifier":
        base_mdl = MLPClassifier(early_stopping=True, n_iter_no_change=50)
    else:
        raise NotImplementedError("model type not implemented")
    if max([len(a) for a in param_dict.values()]) > 1:
        # If there is tuning to do
        grid_cv = GridSearchCV(estimator=base_mdl, param_grid=param_dict, cv=3, n_jobs=1, verbose=4, scoring='neg_log_loss')
        grid_cv.fit(
            data_dict["train"]["X"][:,train_axes],
            data_dict["train"]["y"].flatten(),
        )
        logging.info("CV BEST SCORE %f", grid_cv.best_score_)
        logging.info("CV BEST PARAMS %s", grid_cv.best_params_)
        print(grid_cv.best_params_)
        base_mdl.set_params(**grid_cv.best_params_)
    else:
        param_dict0 = {k: v[0] for k,v in param_dict.items()}
        base_mdl.set_params(**param_dict0)
        print(base_mdl)
        logging.info(base_mdl)
    # print("MODEL OOB", mdl.oob_score_)
    
    if args.model_type != "LogisticRegression":
        mdl = CalibratedClassifierCV(base_mdl, cv=args.calib_cv, method=args.calib_method)
    else:
        mdl = base_mdl

    print("training data %s", data_dict["train"]["X"][:,train_axes].shape)
    logging.info("training data %s", data_dict["train"]["X"][:,train_axes].shape)
    mdl.fit(
        data_dict["train"]["X"][:,train_axes],
        data_dict["train"]["y"].flatten(),
        )
    mdl.train_axes = train_axes
    
    pred_probs = mdl.predict_proba(data_dict["train"]["X"][:,train_axes])[:,1]
    val, cts = np.unique(pred_probs, return_counts=True)
    print("most common count", cts.max())
    print("common value", val[np.argmax(cts)])

    if args.model_type == "LogisticRegression":
        logging.info("intercept %s", mdl.intercept_)
        logging.info("coef %s", mdl.coef_)

    with open(args.mdl_file, "wb") as f:
        pickle.dump(mdl, f)

    # plot calibration curve
    if args.plot_file is not None:
        test_key = "bigtest" if "bigtest" in data_dict else "test"
        pred_prob = mdl.predict_proba(data_dict[test_key]["X"][:,train_axes])[:,1]
        print("pred_prob", pred_prob)

        _, axs = plt.subplots(1, 2)
        calib_disp = CalibrationDisplay.from_estimator(mdl, data_dict[test_key]["X"][:,train_axes], data_dict[test_key]["y"].flatten(), n_bins=10, name=args.model_type, ax=axs[0], strategy='quantile')
        roc_disp = RocCurveDisplay.from_estimator(mdl, data_dict[test_key]["X"][:,train_axes], data_dict[test_key]["y"].flatten(), name=args.model_type, ax=axs[1])
        plt.savefig(args.plot_file)
        print(args.plot_file)
        
        # plt.clf()
        # plt.hist(pred_prob, log="y")
        # plt.show()


if __name__ == "__main__":
    main()
