import os
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Read data")
    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.75
    )
    parser.add_argument(
        "--data-files",
        type=str,
        default="exp_zsfg/zsfg_data_tiny.csv",
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
    args.data_files = args.data_files.split(",")
    args.out_file = args.out_file_template.replace('SEED', str(args.seed))
    args.log_file = args.log_file_template.replace('SEED', str(args.seed))
    return args

def sample_data(dat_df):
    """
    Randomly grab an encounter from each patient (so randomness comes from patients with multiple encounters)
    """
    # Shuffle data
    shuffled_idx = np.random.choice(dat_df.shape[0], size=dat_df.shape[0], replace=False)
    dat_df = dat_df.iloc[shuffled_idx]
    # Grab first encounter per patient in the shuffled list
    dat_df = dat_df.groupby('pat_id_surrogate').first().reset_index()
    print(dat_df)
    return dat_df

def make_sparse_boolean_columns(df):
    """
    Converts columns of a data frame into SparseArrays and returns the data frame with transformed columns.
    :param df: pandas data frame
    """
    print("pre MEM", df.memory_usage().sum())
    converted_cols = []
    for idx, columnName in enumerate(df.columns):
        if df[columnName].dtype != 'boolean':
            continue
        print(idx, columnName)
        df[columnName] = pd.arrays.SparseArray(
            df[columnName].values,
            dtype=bool,
            fill_value=False
        )
        converted_cols.append(columnName[:5])
    print("CONVERTEd", set(converted_cols))
    print("new MEM", df.memory_usage().sum())

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    dat_df = pd.concat([pd.read_parquet(data_file) for data_file in args.data_files])
    logging.info("orig dat_df %s", dat_df.shape)
    np.random.seed(args.seed)

    # Randomly select one encounter per patient
    sampled_df = sample_data(dat_df)
    logging.info("sampled dat_df %s", sampled_df.shape)

    y = sampled_df["y_unplanned_readmission"].to_numpy()
    outcome_cols = [col_name for col_name in sampled_df.columns if col_name.startswith("y_")] + ["contact_date", "pat_id_surrogate"]
    X = sampled_df.iloc[:, ~sampled_df.columns.isin(outcome_cols)]
    make_sparse_boolean_columns(X)
    pat_ids = sampled_df.pat_id_surrogate
    # Feature transformations, imputations, etc
    categorical_transformer = Pipeline(steps=[
        # Encodes the missing data as a separate category
        ('imputer_na', SimpleImputer(strategy='constant', fill_value='missing')),
        ('imputer_unknown', SimpleImputer(strategy='constant', missing_values="Unknown", fill_value='missing')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown="ignore"))])

    transformers = make_column_transformer(
        # TODO: do something smarter for continuous-valued data
        (
            SimpleImputer(strategy="mean"),
            make_column_selector(dtype_include=[np.number]),
        ),
        (
            categorical_transformer,
            make_column_selector(pattern="^demographic*",
                dtype_exclude=[np.number, "boolean"]),
        ),
        (
            categorical_transformer,
            make_column_selector(
                pattern="^index_discharge_*", dtype_exclude=[np.number, "boolean"]
            ),
        ),
        (
            categorical_transformer,
            make_column_selector(pattern="^ept*", dtype_exclude=[np.number, "boolean"]),
        ),
        (
            categorical_transformer,
            make_column_selector(
                pattern="^sdoh_*", dtype_exclude=[np.number, "boolean"]
            ),
        ),
        (
            categorical_transformer,
            make_column_selector(pattern="^lab*", dtype_exclude=[np.number, "boolean"]),
        ),
        n_jobs=1,
        remainder='passthrough',
        verbose=True,
    )
    steps = [
        ("col_transforms", transformers),
    ]

    # Fit imputer
    pipe = Pipeline(steps=steps, verbose=True)
    processed_X = pipe.fit_transform(X)
    feature_names = pipe.get_feature_names_out(X.columns)

    # Randomly allocate patients to train v test
    all_patients = pat_ids.unique()
    logging.info("NUM TOT PATIENTS %d", all_patients.size)
    all_patients = np.random.choice(all_patients, all_patients.size, replace=False)
    train_patients = all_patients[:int(all_patients.size * args.train_frac)]
    test_patients = all_patients[int(all_patients.size * args.train_frac):]
    logging.info("train patients %s", train_patients)
    train_idx = pat_ids.isin(train_patients)
    test_idx = pat_ids.isin(test_patients)
    logging.info("num train %d", train_idx.sum())
    logging.info("num test %d", test_idx.sum())

    with open(args.out_file, "wb") as f:
        pickle.dump({
                "train": {
                    "X": csr_matrix(processed_X[train_idx]),
                    "y": y[train_idx],
                },
                "test": {
                    "X": csr_matrix(processed_X[test_idx]),
                    "y": y[test_idx],
                },
                "feature_names": feature_names,
            },
            f,
        )

if __name__ == "__main__":
    main()
