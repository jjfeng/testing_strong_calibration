import os
import argparse
import logging

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="aggregate feature importance files"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
    )
    parser.add_argument(
        "--result-files",
        type=str,
    )
    parser.add_argument(
        "--test-stat",
        type=str,
        help="which test stat to plot"
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default="_output/feat_import.png",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="_output/res.csv",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="_output/log.txt",
    )
    args = parser.parse_args()
    args.result_files = args.result_files.split("+")
    return args

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    
    all_res = []
    for f in args.result_files:
        try:
            res = pd.read_csv(f)
            all_res.append(res)
        except FileNotFoundError as e:
            logging.info(e)
            continue

    all_res = pd.concat(all_res).reset_index()
    all_res = all_res[all_res.test_stat == args.test_stat]
    all_res['feature'].replace({'orig_pred': 'Prediction'}, inplace=True)
    if (all_res.feature == 'x0').any():
        mask = all_res.feature.str.startswith('x')
        feat_idx = all_res[mask].feature.str.split("x",
                expand=True)[1].astype(int)
        all_res['feature'][mask] = "x" + (feat_idx + 1).astype(str)
        order = ["Prediction"] + [f"x{i + 1}" for i in range(feat_idx.unique().size)]

    print(all_res)

    feat_summ_df = all_res.groupby(['n_test', 'feature']).mean().reset_index()
    feat_summ_df.to_csv(args.csv_file, index=False)
    feat_summ_df = feat_summ_df.rename({'n_test': 'n'}, axis=1)

    # Plot only a subset of the test statistics
    sns.set_context('paper', font_scale=1.8)

    ax = sns.barplot(
        feat_summ_df,
        y="feature",
        x="importance",
        hue="n",
        order=order)
    sns.despine()
    ax.set(xlabel="Importance", ylabel="")
    plt.tight_layout()
    plt.savefig(args.plot_file)

if __name__ == "__main__":
    main()
