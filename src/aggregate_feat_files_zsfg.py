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
    
    all_res = []
    for idx in range(1, 1 + args.num_seeds):
        f = args.result_files.replace('SEED', str(idx))
        try:
            res = pd.read_csv(f)
            res['feature'] = res.feature.str.replace('pipeline-1__demographic_|simpleimputer__demographic_',
                    '', regex=True)
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
        'ethnic_group_Decline to Answer': 'Ethnic group: Decline to Answer',
        'ethnic_group_Yes - Hispanic, Latino/a, or Spanish origin': 'Ethnic group: Hispanic, Latino/a, or Spanish origin',
        'ethnic_group_No - Hispanic, Latino/a, or Spanish origin': 'Ethnic group: Not of Hispanic, Latino/a, or Spanish origin',
        'race1_Native Hawaiian or Other Pacific Islander': 'Race: Native Hawaiian or Other Pacific Islander',
        'race1_American Indian or Alaska Native': 'Race: American Indian or Alaska Native',
        'patient_contacts': 'Number of patient contacts',
        'marital_status_Married': 'Marital status: Married',
        'marital_status_Divorced': 'Marital status: Divorced',
        'marital_status_Single': 'Marital status: Single',
        'marital_status_Widowed': 'Marital status: Widowed',
        'marital_status_missing': 'Marital status: Missing',
        'marital_status_Other': 'Marital status: Other',
        'marital_status_Legally Separated': 'Marital status: Legally Separated',
        'race1_Black or African American': 'Race: Black or African American',
        'race1_Decline to Answer': 'Race: Decline to Answer',
        'race1_Asian': 'Race: Asian',
        'race1_White': 'Race: White',
        'race1_Other': 'Race: Other',
        'race1_missing': 'Race: Missing',
        'sex_Female': 'Sex: Female',
        'sex_Male': 'Sex: Male',
        'sex_Nonbinary': 'Sex: Nonbinary',
        'sex_missing': 'Sex: Missing',
        'sex_Other': 'Sex: Other',
        'sex_X': 'Sex: X',
        'age': 'Age',
        'has_phone': 'Has phone',
        'has_address': 'Has address',
        'ssn_status_missing': 'SSN status: Missing',
        'ssn_status_Patient Has SSN': 'SSN status: Patient Has SSN',
        'ssn_status_Patient has No SSN': 'SSN status: Patient has No SSN',

    }, inplace=True)
    plt.figure(figsize=(10,6))
    sns.set_context('paper', font_scale=2)

    # Select the test statistic
    mask = (feat_summ_df.test_stat == args.test_stat)
    feat_summ_df = feat_summ_df[mask]

    # Plot only the top-most important variables
    threshold = feat_summ_df.importance.sort_values().iloc[-15]

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
