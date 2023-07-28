"""
Code for unit-testing
"""
import unittest

from detector import *

dummy_detector_meta_df = pd.DataFrame({
    'model_idx': [0],
    'class_name': ['test_model'],
    'name': ['test_model']
})

def run_hypo_test_greater(detector, pred_val_list, test_residuals, pred_prob_valid):
    
    test_stats_df, plot_df = detector._get_test_stats(pred_val_list, dummy_detector_meta_df, test_residuals, pred_prob=pred_prob_valid, do_print=True)
    boot_dfs = detector._simulate_null_res(pred_prob_valid, pred_val_list, dummy_detector_meta_df, test_greater=True)
    test_stats_df['pval'] = [np.mean(test_stats_df.test_stat[col_name] <= boot_dfs[col_name]) for col_name in test_stats_df.index]        
    test_stats_df['method'] = test_stats_df.index
    test_stats_df = test_stats_df.merge(dummy_detector_meta_df, how="inner", left_on="selected_model", right_on="model_idx")
    test_stats_df.drop('model_idx', axis=1, inplace=True)

    return test_stats_df, plot_df

def run_hypo_test_twosided(detector, pred_val_list, test_residuals_greater, test_residuals_less, pred_prob_valid):
    test_stats_df, plot_df = detector._get_test_stats_two_sided(pred_val_list, dummy_detector_meta_df, test_residuals_greater, test_residuals_less, pred_prob=pred_prob_valid, do_print=True)
    boot_dfs = detector._simulate_null_res(pred_prob_valid, pred_val_list, dummy_detector_meta_df)
    test_stats_df['pval'] = [np.mean(test_stats_df.test_stat[col_name] <= boot_dfs[col_name]) for col_name in test_stats_df.index]        
    test_stats_df['method'] = test_stats_df.index
    test_stats_df = test_stats_df.merge(dummy_detector_meta_df, how="inner", left_on="selected_model", right_on="model_idx")
    test_stats_df.drop('model_idx', axis=1, inplace=True)

    return test_stats_df, plot_df

def make_oracle_test_data(n, detector):
    beta = 1
    est_beta = 0.5
    x = np.random.uniform(low=-5, high=5, size=n)
    true_mu = 1/(1 + np.exp(-x * beta))
    y = np.random.binomial(n=1, p=true_mu, size=n)
    pred_prob_valid = 1/(1 + np.exp(-x * (beta * (np.abs(x) > 3) + est_beta * (np.abs(x) < 3))))
    # pred_prob_valid = 1/(1 + np.exp(-x * est_beta))
    test_residuals_greater = (y - (pred_prob_valid + detector.tolerance_prob)).reshape((1,-1))
    test_residuals_less = (y - (pred_prob_valid - detector.tolerance_prob)).reshape((1,-1))
    pred_val_list = [detector._combine_pred_vals(
        true_mu - (pred_prob_valid + detector.tolerance_prob),
        true_mu - (pred_prob_valid - detector.tolerance_prob),
        )]
    return pred_val_list, test_residuals_greater, test_residuals_less, pred_prob_valid

N_BOOT = 100
n_reps = 20

class TestDetectors(unittest.TestCase):
    alpha = 0.1
    def test_oracle_greater(self):
        np.random.seed(1)

        score_dfs = []
        chisq_dfs = []
        for i in range(n_reps):
            score_detector = SplitScore(
                {},
                [],
                np.arange(1),
                {"zero_weights": [0]},
                0.5,
                n_boot=N_BOOT,
                tolerance_prob=0,
                alternative="greater",
            )
            chisq_detector = SplitChiSquared(
                {},
                [],
                np.arange(1),
                {"zero_weights": [0]},
                0.5,
                n_boot=N_BOOT,
                tolerance_prob=0,
                alternative="greater",
            )
            pred_val_list, test_residuals, _, pred_prob_valid = make_oracle_test_data(n=1600, detector=score_detector)
            score_test_stats_df, plot_df = run_hypo_test_greater(score_detector, pred_val_list, test_residuals, pred_prob_valid)
            score_test_stats_df['seed'] = i
            score_dfs.append(score_test_stats_df)
            chisq_test_stats_df, _ = run_hypo_test_greater(chisq_detector, pred_val_list, test_residuals, pred_prob_valid)
            chisq_test_stats_df['seed'] = i
            chisq_dfs.append(chisq_test_stats_df)
        
        score_detector.plot(plot_df, "_output/test.png")

        score_dfs = pd.concat(score_dfs)
        chisq_dfs = pd.concat(chisq_dfs)
        score_dfs['reject'] = score_dfs.pval < self.alpha
        chisq_dfs['reject'] = chisq_dfs.pval < self.alpha
        score_power = score_dfs.groupby(['method']).reject.mean()
        chisq_power = chisq_dfs.groupby(['method']).reject.mean()
        print("score power", score_power)
        print("chisq power", chisq_power)
        self.assertGreater(score_power, chisq_power + 0.14)
    
    def test_oracle_twosided(self):
        np.random.seed(1)

        score_dfs = []
        chisq_dfs = []
        for i in range(n_reps):
            score_detector = SplitScoreTwoSided(
                {},
                [],
                np.arange(1),
                {"zero_weights": [0]},
                0.5,
                n_boot=N_BOOT,
                tolerance_prob=0,
                alternative="greater",
            )
            chisq_detector = SplitChiSquaredTwoSided(
                {},
                [],
                np.arange(1),
                {"zero_weights": [0]},
                0.5,
                n_boot=N_BOOT,
                tolerance_prob=0,
                alternative="greater",
            )
            pred_val_list, test_residuals_greater, test_residuals_less, pred_prob_valid = make_oracle_test_data(n=100, detector=score_detector)
            
            score_test_stats_df, plot_df = run_hypo_test_twosided(score_detector, pred_val_list, test_residuals_greater, test_residuals_less, pred_prob_valid)
            score_test_stats_df['seed'] = i
            score_dfs.append(score_test_stats_df)
            chisq_test_stats_df, _ = run_hypo_test_twosided(chisq_detector, pred_val_list, test_residuals_greater, test_residuals_less, pred_prob_valid)
            chisq_test_stats_df['seed'] = i
            chisq_dfs.append(chisq_test_stats_df)
        
        score_detector.plot(plot_df, "_output/test.png")

        score_dfs = pd.concat(score_dfs)
        chisq_dfs = pd.concat(chisq_dfs)
        score_dfs['reject'] = score_dfs.pval < self.alpha
        chisq_dfs['reject'] = chisq_dfs.pval < self.alpha
        print(score_dfs)
        print(chisq_dfs)
        score_power = score_dfs.groupby(['method']).reject.mean()
        chisq_power = chisq_dfs.groupby(['method']).reject.mean()
        print("score power", score_power)
        print("chisq power", chisq_power)
        self.assertGreater(score_power.max_vec, chisq_power.max() + 0.14)
