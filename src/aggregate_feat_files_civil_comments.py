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
    return args

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    if args.num_seeds == 0:
        return
    
    all_res = []
    for idx in range(1, 1 + args.num_seeds):
        f = args.result_files.replace('SEED', str(idx))
        try:
            res = pd.read_csv(f)
            res['feature'] = res.feature.str.replace('demographic_', '', regex=True)
            all_res.append(res)
        except FileNotFoundError as e:
            logging.info(e)
            continue

    all_res = pd.concat(all_res).reset_index()
    print(all_res)

    feat_summ_df = all_res.groupby(['test_stat', 'feature']).mean().reset_index()
    print(feat_summ_df)
    feat_summ_df.to_csv(args.csv_file, index=False)

    # Plot only a subset of the test statistics
    feat_summ_df['feature'].replace({
        'orig_pred':'Prediction',
        'black':'Black',
        'christian':'Christian',
        'muslim': 'Muslim',
        'homosexual_gay_or_lesbian': 'Homosexual',
        'transgender': 'Transgender',
        'other_sexual_orientation': 'Other sexual orientation',
        'other_gender': 'Other gender',
        'asian': 'Asian',
        'intellectual_or_learning_disability': 'Intellectual disability',
    }, inplace=True)
    feat_summ_df = feat_summ_df[np.logical_not(feat_summ_df['feature'].str.startswith('embedding'))]

    #plt.figure(figsize=(10,6))
    sns.set_context('paper', font_scale=2)

    # Select the test statistic
    mask = (feat_summ_df.test_stat == args.test_stat)
    feat_summ_df = feat_summ_df[mask]

    # Plot only the top-most important variables
    threshold = feat_summ_df.importance.sort_values().iloc[-10]

    feat_summ_df = feat_summ_df[feat_summ_df.importance >= threshold]

    ax = sns.barplot(
        feat_summ_df,
        y="feature",
        x="importance",
        order=feat_summ_df.sort_values('importance').feature)
    ax.set(xlabel="Importance", ylabel="")
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.plot_file)

if __name__ == "__main__":
    main()
