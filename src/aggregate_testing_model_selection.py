import os
import argparse
import logging

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="aggregate result files, get model selected"
    )
    parser.add_argument(
        "--result-files",
        type=str,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--plot-x",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plot-row",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="_output/res.csv",
    )
    parser.add_argument(
        "--plot-model-file",
        type=str,
        default="_output/selected_model.png",
    )
    parser.add_argument(
        "--plot-stat-file",
        type=str,
        default="_output/selected_stat.png",
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
            res = pd.read_csv(f, index_col=0)
            # if res.pval.min() < args.alpha:
            all_res.append(res)
        except FileNotFoundError as e:
            print(e)
            continue

    all_res = pd.concat(all_res).reset_index(drop=True)
    all_res['detection_test'] = all_res.detector.str.split("_", expand=True)[0]
    all_res = all_res[(all_res.method == "max_gval") & all_res.detection_test.isin(["CVScoreTwoSided","SplitScoreTwoSided"])].reset_index(drop=True)

    print("plotting model select")
    all_res_mdl = all_res[['detector', 'detection_test', args.plot_x,
        args.plot_row, 'class_name']].groupby(by=['detector', 'detection_test', args.plot_x, args.plot_row]).value_counts().reset_index().rename({0:'counts'}, axis=1)
    logging.info(all_res_mdl)
    sns.set_context('paper', font_scale = 1.8)
    all_res_mdl = all_res_mdl.rename({
        'tolerance': 'Tolerance',
        'class_name': 'Detector'
    }, axis=1)
    ax = sns.relplot(
        all_res_mdl,
        x="n_test",
        y="counts",
        style="Detector",
        row="detection_test",
        hue="Tolerance",
        kind="line",
        linewidth=3)
    # update to pretty axes and titles
    ax.set_axis_labels(
        "n",
        "Count"
        ).set_titles(
            ""
        )
    plt.savefig(args.plot_model_file)

    # print("plotting test stat select")
    # all_stat_mdl = all_res[['detector', 'detection_test', args.plot_x, args.plot_row, 'selected_z_func']].groupby(by=['detector', 'detection_test', args.plot_x, args.plot_row]).value_counts().reset_index().rename({0:'counts'}, axis=1)
    # logging.info(all_stat_mdl)
    # sns.relplot(
    #     all_stat_mdl,
    #     x="n_test",
    #     y="counts",
    #     hue="selected_z_func",
    #     row="detection_test",
    #     style="tolerance",
    #     kind="line")
    # plt.savefig(args.plot_stat_file)

if __name__ == "__main__":
    main()
