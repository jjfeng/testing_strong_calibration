import os
import logging
import argparse

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

PLOT_STATS = ['last_gval', 'max_gval', 'hl_2', 'hl_10', 'adaptchisq_2', 'adaptchisq_10']
PLOT_DICT = {
    'CVScoreTwoSided_KernelLogistic_RandomForestRegressor_last_gval': 'AdaptiveScoreSimple',
    'CVScoreTwoSided_RandomForestRegressor_KernelLogistic_last_gval': 'AdaptiveScoreSimple',
    'CVScore_RandomForestRegressor_KernelLogistic_last_gval': 'AdaptiveScoreSimple',
    'CVScoreTwoSided_KernelLogistic_RandomForestRegressor_max_gval': 'AdaptiveScoreCUSUM',
    'CVScoreTwoSided_RandomForestRegressor_KernelLogistic_max_gval': 'AdaptiveScoreCUSUM',
    'CVScore_RandomForestRegressor_KernelLogistic_max_gval': 'AdaptiveScoreCUSUM',
    'LogisticRecalibrationTwoSided_None_max_gval': 'PrespecScore',
    'LogisticRecalibration_None_max_gval': 'PrespecScore',
    'HosmerLemeshow_None_hl_2': 'PrespecChiSq (bins=2)',
    'HosmerLemeshow_None_hl_10': 'PrespecChiSq (bins=10)',
    'CVChiSquaredTwoSided_KernelLogistic_RandomForestRegressor_adaptchisq_2': 'AdaptChiSq (bins=2)',
    'CVChiSquared_RandomForestRegressor_KernelLogistic_adaptchisq_2': 'AdaptChiSq (bins=2)',
    'CVChiSquaredTwoSided_KernelLogistic_RandomForestRegressor_adaptchisq_10': 'AdaptChiSq (bins=10)',
    'CVChiSquared_RandomForestRegressor_KernelLogistic_adaptchisq_10': 'AdaptChiSq (bins=10)',
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="concatenate csvs"
    )
    parser.add_argument(
        "--type-i-error",
        action="store_true",
    )
    parser.add_argument(
        "--x-ticks",
        type=str,
        default="400+800+1200+1600",
    )
    parser.add_argument(
        "--proposed-only",
        action="store_true",
    )
    parser.add_argument(
        "--extra-label-name",
        type=str,
    )
    parser.add_argument(
        "--extra-label-val",
        type=str,
    )
    parser.add_argument(
        "--result-files",
        type=str,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="log.txt",
    )
    parser.add_argument(
        "--tex-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--csv-file",
        type=str,
    )
    parser.add_argument(
        "--do-agg",
        action="store_true",
    )
    parser.add_argument(
        "--plot-x",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plot-y",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plot-col",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plot-row",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plot-style",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plot-hue",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    args.result_files = args.result_files.split("+")
    args.x_ticks = list(map(int, args.x_ticks.split("+")))
    return args

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )

    all_res = []
    for idx, f in enumerate(args.result_files):
        res = pd.read_csv(f)
        print("res.detector", res.detector)
        all_res.append(res)
    all_res = pd.concat(all_res)
    if args.extra_label_val:
        all_res[args.extra_label_name] = args.extra_label_val
    all_res = all_res.reset_index(drop=True)

    all_res.to_csv(args.csv_file, index=False)
    if args.plot_file is not None:
        all_res['detector_method'] = all_res.detector + "_" + all_res.method
        all_res['detector_method'] = all_res['detector_method'].replace(PLOT_DICT)
        all_res = all_res.rename({'detector_method': 'Test'}, axis=1)
        all_res['alternative'] = all_res['alternative'].replace({'less':
            'over-estimates', 'greater': 'under-estimates', 'both': 'both'})
        
        plt.figure(figsize=(10,5))
        sns.set_context('paper', font_scale=1.8)
        if args.proposed_only:
            plot_res = all_res[all_res.Test == "AdaptiveScoreCUSUM"]
        else:
            plot_res = all_res[all_res.Test.isin(list(PLOT_DICT.values()))]
        print(plot_res)
        ax = sns.relplot(
            data=plot_res,
            x=args.plot_x,
            y=args.plot_y,
            hue="Test" if args.plot_hue == "detector_method" else args.plot_hue,
            col=args.plot_col,
            row=args.plot_row,
            kind="line",
            markers=True,
            linewidth=3,
            legend=not args.proposed_only,
        )
        for g in ax.axes.flat:
            g.set_xticks(args.x_ticks)
        # update to pretty axes and titles
        ax.set_axis_labels(
            "n",
            "Type I error" if args.type_i_error else "Power"
            ).set_titles(
                "Tolerance: {col_name}" if args.plot_col == 'tolerance' else
                'Alternative: {col_name}'
            ).set(ylim=(0,1))
        plt.savefig(args.plot_file)

    if args.tex_file is not None:
        power_df = all_res.groupby([args.plot_x, args.plot_row, args.plot_col,'detector', 'method']).agg({
            'reject': ['mean', 'count'],
            'pval': ['mean'],
        }).reset_index()
        print(power_df)
        power_df.columns = [' '.join(col).strip() for col in power_df.columns.values]
        logging.info(power_df)
        power_df.to_latex(args.tex_file, index=False)

if __name__ == "__main__":
    main()
