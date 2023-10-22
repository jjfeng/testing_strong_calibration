import os
import argparse
import logging

import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="aggregate result files, get power of methods"
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
        "--detector-name",
        type=str,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--do-agg",
        action="store_true",
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
    parser.add_argument(
        "--pval-plot",
        type=str,
        default=None,
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
            res['seed'] = idx
            all_res.append(res)
        except FileNotFoundError as e:
            print(e)
            continue

    all_res = pd.concat(all_res).reset_index()
    if 'pval' in all_res.columns:
        all_res['reject'] = (all_res.pval <= args.alpha).astype(int)
        logging.info("mean pval %s", all_res.groupby('method').mean())

    if args.do_agg:    
        power_df = all_res.groupby('method').agg({
            'reject': ['mean', 'count'],
        }).reset_index()
        power_df.columns = [' '.join(col).strip() for col in power_df.columns.values]
        logging.info("power %s", power_df)
        
        print(power_df)
        power_df['detector'] = args.detector_name
        power_df.to_csv(args.csv_file, index=False)
    else:
        all_res['detector'] = args.detector_name
        all_res.to_csv(args.csv_file, index=False)

    if args.pval_plot:
        import statsmodels.api as sm
        print(np.sort(all_res[all_res.method == "maxw_vec"].pval))
        print(np.sort(all_res[all_res.method == "max_vec"].pval))
        pplot = sm.ProbPlot(all_res[all_res.method == "max_vec"].pval, dist=stats.uniform)
        fig = pplot.qqplot(line="45")
        # stats.probplot(all_res[all_res.method == "maxw_vec"].pval, dist=stats.uniform, plot=plt)
        plt.savefig(args.pval_plot)

if __name__ == "__main__":
    main()
