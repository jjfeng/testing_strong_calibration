import logging
from typing import List, Dict

import sklearn
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, KFold, ParameterGrid

from common import make_prob, to_safe_logit

LINEAR_MODELS = ["Lasso", "LogisticRegression", "KernelLogistic"]
CLASSIFIERS = ["LogisticRegression", "KernelLogistic"]
REGRESSORS = ["Lasso", "LinearRegression", "RandomForestRegressor"]

class SubgroupDetector:
    """
    Tests if there is a subgroup for which a ML algorithm is poorly calibrated
    """
    def _get_residual(self, np_Y, pred_prob, test_greater):
        """
        @return the residual compared to the shifted predicted probability, according to the tolerance
            and whether we are testing if the true prob is greater or less
        """
        test_sign = 1 if test_greater else -1
        assert (pred_prob.max() < 1) and (pred_prob.min() > 0)
        null_prob = make_prob(pred_prob + test_sign * self.tolerance_prob)
        residual = test_sign * (np_Y - null_prob)
        return residual

    def test(
        self,
        np_X: np.ndarray,
        np_Y: np.ndarray,
        ml_mdl,
    ):
        """
        @param np_X: variables in audit data
        @param np_Y: outcomes in audit data
        @param ml_mdl: the ML algorithm to audit, should be a sklearn model

        @return test results in a pandas dataframe, additional data frame that can be used for plotting,
            if the plotting function is implemented
        """
        plot_df = None
        if self.one_sided:
            if self.alternative == "greater":
                result_df, plot_df = self._test_one_sided(np_X, np_Y, ml_mdl, test_greater=True)
            elif self.alternative == "less":
                result_df, plot_df = self._test_one_sided(np_X, np_Y, ml_mdl, test_greater=False)
            else:
                # If testing two-sided but the test procedure only does one-sided, do a Bonferroni correction
                res_greater_df, _ = self._test_one_sided(np_X, np_Y, ml_mdl, test_greater=True)
                res_less_df, _ = self._test_one_sided(np_X, np_Y, ml_mdl, test_greater=False)
                if res_less_df is not None and res_greater_df is not None:
                    result_df = pd.concat([res_greater_df, res_less_df]).reset_index(drop=True)
                    print("result_df", result_df)
                    if 'pval' in res_greater_df.columns:
                        print(result_df.groupby('method').pval.idxmin())
                        result_df = result_df.loc[result_df.groupby('method').pval.idxmin()].reset_index(drop=True)
                        # Adjust for multiple comparisons
                        result_df['pval'] = result_df.pval * 2
                    else:
                        raise ValueError('confused by the set of columns available')
                elif res_less_df is None and res_greater_df is not None:
                    result_df = res_greater_df
                elif res_less_df is not None and res_greater_df is None:
                    result_df = res_less_df
        else:
            assert self.alternative == 'both'
            result_df, plot_df = self._test_two_sided(np_X, np_Y, ml_mdl)

        return result_df, plot_df

    def _test_one_sided(
        self, np_X: np.ndarray, np_Y: np.ndarray, ml_mdl, test_greater: bool=True
    ):
        """
        Performs the one-sided test. Calculates the critical value using Monte Carlo procedure to
        control finite-sample Type I error.

        @param test_greater: are we testing if the true probabilities are greater than predicted
        """
        pred_prob = ml_mdl.predict_proba(np_X)[:, 1]

        # Compute test statistic
        test_stats_df = self._calc_test_stats(
            np_X, np_Y, pred_prob, test_greater
        )
        print("test statistics", test_stats_df)
        logging.info("test statistics")
        logging.info(test_stats_df)
        
        # Simulate the test statistic under the worst-case null distribution
        boot_dfs = []
        for i in range(self.n_boot):
            perturbed_mu = make_prob(pred_prob + self.tolerance_prob * (1 if test_greater else -1))
            sampled_y = np.random.binomial(n=1, p=perturbed_mu, size=pred_prob.size)
            boot_df = self._calc_test_stats(
                np_X, sampled_y, pred_prob, test_greater
            )
            boot_dfs.append(boot_df)
        boot_dfs = pd.concat(boot_dfs)

        # Summarize results
        pvals = []
        for col_name in test_stats_df.index:
            pvals.append(pd.DataFrame({
                "method": [col_name],
                "pval": [np.mean(test_stats_df[col_name] <= boot_dfs[col_name])]
            }))

        return pd.concat(pvals), None
    
    def plot(self, plot_dfs, plot_file: str):
        # Not implemented by default
        return
    
    def get_feat_imports(self, orig_ml_mdl, test_X, test_Y) -> pd.DataFrame:
        # Not implemented by default
        return

class HosmerLemeshow(SubgroupDetector):
    """
    Squared subgroup-level residuals along the predicted logit (or covariate axes)
    """
    main_statistic = "hl"
    one_sided = True

    def __init__(self, axes: List[int], n_boot: int, tolerance_prob: float, alternative: str, n_bin_list=[2,4,8,10]):
        """
        @param axes: a list of which pre-defined axes to bin the data and perform an H-L test.
                    -1 means to bin along the predicted probabilities
        @param n_boot: number of bootstrap samples
        @param tolerance_prob: probability within tolerance rance
        @param alternative: greater, less, or both
        @param n_bin_list: number of bins to consider for performing test
        """
        self.axes = axes
        self.n_boot = n_boot
        self.tolerance_prob = tolerance_prob
        self.alternative = alternative
        self.n_bin_list = n_bin_list
        
    def _get_group_normalizer(self, pred_prob_sorted, split_idxs):
        """
        @return the variance of the sum within each group
        """
        return np.array([
            np.sum(pred_prob_sorted[grp_idxs] * (1 - pred_prob_sorted[grp_idxs]))
            for grp_idxs in split_idxs])

    def _calc_test_stats(
        self, np_X, np_Y, pred_prob, test_greater=True
    ) -> pd.Series:
        residual = self._get_residual(np_Y, pred_prob, test_greater)

        discriminating_indices = []
        sort_options = []
        
        if -1 in self.axes:
            discriminating_indices.append(pred_prob)
            sort_options += [
                np.argsort(pred_prob),
            ]
        for ax in self.axes:
            if ax >= 0:
                discriminating_indices.append(np_X[:,ax])
                sort_options += [
                    np.argsort(np_X[:,ax]),
                ]
        
        hl_stat_dict = {}
        for n_bins in self.n_bin_list:
            hl_statistics = []
            for sort_option in sort_options:
                test_residual_sorted = residual[sort_option]
                pred_prob_sorted = pred_prob[sort_option]
                split_idxs = np.array_split(np.arange(test_residual_sorted.size), n_bins)
                grp_normalizer = self._get_group_normalizer(pred_prob_sorted, split_idxs)
                hl_cells = np.array([
                    np.sum(test_residual_sorted[grp_idxs])
                    for grp_idxs in split_idxs])
                hl_normalized_cells = np.power(np.maximum(0, hl_cells), 2)/grp_normalizer
                hl_statistics.append(np.sum(hl_normalized_cells))
            hl_stat_dict["hl_%d" % n_bins] = np.max(hl_statistics)
        
        test_stats = pd.Series(hl_stat_dict)
        return test_stats

class SplitTesting(SubgroupDetector):
    """
    Sample splitting for testing subgroups, only does one-sided
    (Must implement the actual test statistics)
    """
    one_sided = True
    def __init__(
        self,
        detection_mdl_dict: Dict,
        feature_names: List[str],
        axes: np.ndarray,
        param_dict: Dict,
        split: float,
        n_boot: int,
        tolerance_prob: float,
        alternative: str,
        random_state: int = 0,
    ):
        """
        @param detector_mdl_dict: A dictionary mapping string (model name) to actual sklearn base model
        @param feature_names: solely for plotting later
        @param axes: which variables to include for constructing candidate subgroups
        @param param_dict: dict mapping model name to which hyperparameters to search over
        @param split: proportion of data to dedicate to training
        @param n_boot: number of bootstrap samples for computing p-value
        @param tolerance_prob: tolerance for strong calibration test
        @param alternative: greater or less
        @param random_state: seed for tests
        """
        self.split = split
        self.zero_weights = param_dict["zero_weights"]
        self.param_grid_dict = {mdl_name: ParameterGrid(mdl_param_dict) for mdl_name, mdl_param_dict in param_dict.items() if mdl_name != "zero_weights"}
        self.detection_base_mdl_dict = detection_mdl_dict
        self.axes = axes
        self.feature_names = feature_names
        self.n_boot = n_boot
        self.tolerance_prob = tolerance_prob
        self.alternative = alternative
        self.random_state = random_state

    def _get_x_features(self, np_X, pred_prob):
        """
        Which features to extract for predicting residuals
        """
        pred_logit = np.log(pred_prob/(1 - pred_prob)).reshape((-1,1))
        return np.concatenate([pred_logit, np_X[:, self.axes]], axis=1)
    
    def _get_pred_vals(self, residual_mdl, np_X_aug, pred_prob, detector_name, test_greater):
        if detector_name in CLASSIFIERS:
            # If a classification model
            test_sign = 1 if test_greater else -1
            null_prob = make_prob(pred_prob + test_sign * self.tolerance_prob)
            pred_vals = test_sign * (residual_mdl.predict_proba(np_X_aug)[:,1] - null_prob)
        else:
            # If a regression model
            pred_vals = residual_mdl.predict(np_X_aug) 
        return pred_vals

    def _fit_detectors(
        self, np_X, np_Y, pred_prob, train_idxs, test_greater=True,
    ):
        """
        Train model to predict residuals and test on held out data
        """
        residual = self._get_residual(np_Y, pred_prob, test_greater)
        
        # Train the residual models
        np_X_aug = self._get_x_features(np_X, pred_prob)
        all_residual_models = []
        model_descript_dicts = []
        zero_X = np.concatenate([np_X_aug[train_idxs,:1], np.random.uniform(low=-5, high=5, size=(train_idxs.size,np_X_aug.shape[1] - 1))], axis=1)
        sample_weights = np.ones(train_idxs.size)
        for detection_mdl_name, detection_base_mdl in self.detection_base_mdl_dict.items():
            mdl_idx = 0
            for param_dict in self.param_grid_dict[detection_mdl_name]:
                for zero_weight in (self.zero_weights if detection_mdl_name in REGRESSORS else [0]):
                    print(detection_mdl_name, param_dict, zero_weight)
                    # try:
                    detector = sklearn.base.clone(detection_base_mdl)
                    detector.set_params(**param_dict)
                    if detection_mdl_name in CLASSIFIERS:
                        if detection_mdl_name == "KernelLogistic":
                            detector.fit(
                                np_X_aug[train_idxs],
                                np_Y[train_idxs],
                                lr__sample_weight=sample_weights
                            )
                        else:
                            detector.fit(
                                np_X_aug[train_idxs],
                                np_Y[train_idxs],
                                sample_weight=sample_weights
                            )
                    elif detection_mdl_name in REGRESSORS:
                        X_zero_aug = np.concatenate([np_X_aug[train_idxs], zero_X], axis=0)
                        residual_aug = np.concatenate([residual[train_idxs], -np.ones(train_idxs.shape) * self.tolerance_prob], axis=0)
                        detector.fit(
                            X_zero_aug,
                            residual_aug,
                            sample_weight=np.concatenate([sample_weights, np.ones(train_idxs.size) * zero_weight])
                        )
                    else:
                        raise ValueError("detection model not recognized")
                    
                    new_param_dict = param_dict.copy()
                    new_param_dict["zero_weight"] = zero_weight
                    model_descript_dicts.append({
                        "name": f"{detection_mdl_name}_{mdl_idx}",
                        "class_name": detection_mdl_name,
                        "params": new_param_dict})
                    all_residual_models.append(detector)

                    mdl_idx += 1
        
        model_meta_df = pd.DataFrame.from_dict(model_descript_dicts)
        model_meta_df["model_idx"] = model_meta_df.index
        return all_residual_models, model_meta_df
    
    def _get_pred_val_list(self, detectors, detector_meta_df, np_X, pred_prob, valid_idxs, test_greater):
        """
        Get the predictions from all the detectors
        """
        np_X_aug = self._get_x_features(np_X, pred_prob)
        pred_val_list = []
        for idx, detector in enumerate(detectors):
            pred_vals = self._get_pred_vals(detector, np_X_aug[valid_idxs], pred_prob[valid_idxs], detector_meta_df.class_name[idx], test_greater)
            pred_val_list.append(pred_vals)
        
        return pred_val_list
    
    def _get_train_valid_idxs(self, np_Y):
        """
        Split the data
        """
        idxs = np.arange(np_Y.size)
        n_train = int(self.split * np_Y.size)
        train_idxs = idxs[:n_train]
        valid_idxs = idxs[n_train:]
        return train_idxs, valid_idxs

    def _simulate_null_res(self, pred_prob_valid, pred_val_list, detector_meta_df, test_greater=True):
        """
        Simulate the test statistic under the worst-case null distribution
        @return simulated test statistics under the null
        """
        logging.info("SIMULATION NULL ONE SIDED")
        test_sign = 1 if test_greater else -1
        perturbed_mu = make_prob(pred_prob_valid + self.tolerance_prob * test_sign)
    
        sampled_y = np.random.binomial(n=1, p=perturbed_mu, size=(self.n_boot, perturbed_mu.size))
        residuals = self._get_residual(sampled_y, pred_prob_valid, test_greater)
        boot_dfs, _ = self._get_test_stats(pred_val_list, detector_meta_df, residuals, null_prob=perturbed_mu, pred_prob=pred_prob_valid, test_greater=test_greater)
        boot_dfs = boot_dfs.reset_index(drop=True)
        boot_dfs = boot_dfs.pivot(index='method', columns='replicate', values='test_stat')
        return boot_dfs.transpose()

    def _test_one_sided(
        self, np_X, np_Y, ml_mdl, test_greater=True
    ):
        """
        Run test, get test statistic
        """
        np.random.seed(self.random_state)
        pred_prob = ml_mdl.predict_proba(np_X)[:, 1]
        train_idxs, valid_idxs = self._get_train_valid_idxs(np_Y)
        residual = self._get_residual(np_Y, pred_prob, test_greater).reshape((1,-1))
        
        # Fit the detectors
        self.detectors, self.detector_meta_df = self._fit_detectors(
            np_X, np_Y, pred_prob, train_idxs, test_greater=test_greater
        )
        # Get the predictions from all the detectors
        pred_val_list = self._get_pred_val_list(self.detectors, self.detector_meta_df, np_X, pred_prob, valid_idxs, test_greater)
        # Get test statistic for the observed data
        test_stats_df, plot_df = self._get_test_stats(pred_val_list, self.detector_meta_df, residual[:,valid_idxs], pred_prob=pred_prob[valid_idxs], do_print=True)

        print(test_stats_df)
        logging.info("test statistics %s", test_stats_df)
        
        # Simulate the test statistic under the worst-case null distribution
        pred_prob_valid = pred_prob[valid_idxs]
        boot_dfs = self._simulate_null_res(pred_prob_valid, pred_val_list, self.detector_meta_df, test_greater)

        # Summarize results
        test_stats_df['pval'] = [np.mean(test_stats_df.test_stat[col_name] <= boot_dfs[col_name]) for col_name in test_stats_df.index]        
        test_stats_df['method'] = test_stats_df.index
        test_stats_df = test_stats_df.merge(self.detector_meta_df, how="inner", left_on="selected_model", right_on="model_idx")
        test_stats_df.drop('model_idx', axis=1, inplace=True)
        
        return test_stats_df, plot_df
    
class SplitTestingTwoSided(SplitTesting):
    """
    Sample splitting for subgroup testing two-sided
    """
    one_sided = False

    def _combine_pred_vals(self, pred_vals_greater, pred_vals_less):
        """
        Combine the predictions from the fitted models for the residuals estimating E[Y-p_delta(X)|X]
        and E[p_-delta(X) - Y|X] into a single prediction for the largest residuals from the tolerance region.
        """
        pred_vals_greater[pred_vals_greater < 0] = 0
        pred_vals_less[pred_vals_less > 0] = 0

        all_pred_vals = np.concatenate([pred_vals_greater.reshape((-1,1)), pred_vals_less.reshape((-1,1))], axis=1)
        max_idx_pred_vals = np.argmax(np.abs(all_pred_vals), axis=1)
        pred_vals = all_pred_vals[np.arange(all_pred_vals.shape[0]), max_idx_pred_vals]
        return pred_vals

    def _get_pred_vals(self, detectors, X_feats, pred_prob=None, detector_name=None):
        """
        Get predicted values g(X) from the detectors
        Returns:
            predictions in np.array
        """
        detector_greater = detectors[0]
        detector_less = detectors[1]
        if detector_name in CLASSIFIERS:
            null_prob_greater = pred_prob + self.tolerance_prob
            pred_vals_greater = detector_greater.predict_proba(X_feats)[:,1] - null_prob_greater.flatten()
            
            null_prob_less = pred_prob - self.tolerance_prob
            pred_vals_less = detector_less.predict_proba(X_feats)[:,1] - null_prob_less.flatten()
        elif detector_name in REGRESSORS:
            pred_vals_greater = detector_greater.predict(X_feats)
            pred_vals_less = detector_less.predict(X_feats)
        else:
            raise ValueError("detector name not recognized")

        pred_vals = self._combine_pred_vals(pred_vals_greater, pred_vals_less)
        return pred_vals

    def _get_pred_val_list(self, detectors, detector_meta_df, np_X, pred_prob, valid_idxs):
        """
        Get the predictions from all the detectors
        """
        np_X_aug = self._get_x_features(np_X, pred_prob)
        pred_val_list = []
        for idx, detector in enumerate(detectors):
            pred_vals = self._get_pred_vals(detector, np_X_aug[valid_idxs], pred_prob[valid_idxs], detector_meta_df.class_name[idx])
            pred_val_list.append(pred_vals)
        
        return pred_val_list

    def _fit_detectors(
        self, np_X, np_Y, pred_prob, train_idxs
    ):
        """
        Train model to predict residuals and test on held out data.
        Actually trains two models to get the residuals to each edge of the tolerance region
        """
        residuals_greater = self._get_residual(np_Y, pred_prob, test_greater=True)
        residuals_less = -self._get_residual(np_Y, pred_prob, test_greater=False)

        # Train the residual models
        np_X_aug = self._get_x_features(np_X, pred_prob)
        all_residual_models = []
        model_descript_dicts = []
        sample_weights = np.ones(train_idxs.size)
        for detection_mdl_name, detection_base_mdl in self.detection_base_mdl_dict.items():
            mdl_idx = 0
            for param_idx, param_dict in enumerate(self.param_grid_dict[detection_mdl_name]):
                for zero_weight in (self.zero_weights if detection_mdl_name in REGRESSORS else [0]):
                    try:
                        detector_greater = sklearn.base.clone(detection_base_mdl)
                        detector_greater.set_params(**param_dict)
                        detector_less = sklearn.base.clone(detection_base_mdl)
                        detector_less.set_params(**param_dict)
                        
                        if detection_mdl_name in CLASSIFIERS:
                            if detection_mdl_name == "KernelLogistic":
                                detector_greater.fit(
                                    np_X_aug[train_idxs],
                                    np_Y[train_idxs],
                                    lr__sample_weight=sample_weights
                                )
                            else:
                                detector_greater.fit(
                                    np_X_aug[train_idxs],
                                    np_Y[train_idxs],
                                    sample_weight=sample_weights
                                )
                            detector_less = detector_greater
                        elif detection_mdl_name in REGRESSORS:
                            zero_X = np.concatenate([np_X_aug[train_idxs,:1], np.random.uniform(low=-5, high=5, size=(train_idxs.size,np_X_aug.shape[1] - 1))], axis=1)
                            X_zero_aug = np.concatenate([np_X_aug[train_idxs], zero_X], axis=0)
                            residuals_greater_aug = np.concatenate([residuals_greater[train_idxs], self.tolerance_prob * -np.ones(train_idxs.shape)], axis=0)
                            residuals_less_aug = np.concatenate([residuals_less[train_idxs], self.tolerance_prob * np.ones(train_idxs.shape)], axis=0)

                            detector_greater.fit(
                                X_zero_aug,
                                residuals_greater_aug,
                                sample_weight=np.concatenate([np.ones(train_idxs.size), np.ones(train_idxs.size) * zero_weight])
                            )

                            detector_less.fit(
                                X_zero_aug,
                                residuals_less_aug,
                                sample_weight=np.concatenate([np.ones(train_idxs.size), np.ones(train_idxs.size) * zero_weight])
                            )
                        else:
                            raise ValueError("detector name not recognized")
                        
                        new_param_dict = param_dict.copy()
                        new_param_dict["zero_weight"] = zero_weight
                        print("MAKING", detection_mdl_name)
                        model_descript_dicts.append({
                            "name": f"{detection_mdl_name}_{mdl_idx}",
                            "class_name": detection_mdl_name,
                            "params": new_param_dict})
                        logging.info(new_param_dict)
                        mdl_idx += 1

                        all_residual_models.append([detector_greater, detector_less])
                    except ValueError as e:
                        print("FAILED", param_dict)
                        logging.info(e)
                        print(e)
                        1/0
        
        model_meta_df = pd.DataFrame.from_dict(model_descript_dicts)
        model_meta_df["model_idx"] = model_meta_df.index
        return all_residual_models, model_meta_df
    
    def _simulate_null_res(self, pred_prob_valid, pred_val_list, detector_meta_df):
        """Simulate the test statistic under the worst-case null distribution
        Does this by sampling from a uniform distribution first,
        then creates worst-case outcomes

        Returns:
            pandas dataframe with bootstrapped test statistics
        """
        logging.info("SIMULATION NULL TWO SIDED")
        perturbed_mu_greater = make_prob(pred_prob_valid + self.tolerance_prob)
        perturbed_mu_less = make_prob(pred_prob_valid - self.tolerance_prob)
        
        sampled_unif = np.random.uniform(size=(self.n_boot, pred_prob_valid.size))
        residuals_greater = self._get_residual(sampled_unif < perturbed_mu_greater, pred_prob_valid, test_greater=True)
        residuals_less = -self._get_residual(sampled_unif < perturbed_mu_less, pred_prob_valid, test_greater=False)
        boot_dfs, _ = self._get_test_stats_two_sided(
            pred_val_list,
            detector_meta_df,
            residuals_greater,
            residuals_less,
            pred_prob=pred_prob_valid)
        boot_dfs = boot_dfs.reset_index(drop=True)
        boot_dfs = boot_dfs.pivot(index='method', columns='replicate', values='test_stat')
        return boot_dfs.transpose()

    def _test_two_sided(
        self, np_X, np_Y, base_lr
    ):
        logging.info("SPLIT TWO TESTING")
        np.random.seed(self.random_state)
        pred_prob = base_lr.predict_proba(np_X)[:, 1]
        train_idxs, valid_idxs = self._get_train_valid_idxs(np_Y)

        residuals_greater = self._get_residual(np_Y, pred_prob, test_greater=True).reshape((1,-1))
        residuals_less = -self._get_residual(np_Y, pred_prob, test_greater=False).reshape((1,-1))
        
        # fit detectors
        self.detectors, self.detector_meta_df = self._fit_detectors(
            np_X, np_Y, pred_prob, train_idxs
        )
        # get all predictions
        pred_val_list = self._get_pred_val_list(self.detectors, self.detector_meta_df, np_X, pred_prob, valid_idxs)

        # get test statistic
        test_stats_df, plot_df = self._get_test_stats_two_sided(
                pred_val_list,
                self.detector_meta_df,
                residuals_greater[:,valid_idxs],
                residuals_less[:,valid_idxs],
                pred_prob=pred_prob[valid_idxs],
                do_print=True)
        logging.info("test stat %s", test_stats_df)
        print("test statistics", test_stats_df)
        
        # Simulate the test statistic under the worst-case null distribution
        pred_prob_valid = pred_prob[valid_idxs]
        boot_dfs = self._simulate_null_res(pred_prob_valid, pred_val_list, self.detector_meta_df)

        # Summarize results
        test_stats_df['pval'] = [np.mean(test_stats_df.test_stat[col_name] <= boot_dfs[col_name]) for col_name in test_stats_df.index]        
        test_stats_df['method'] = test_stats_df.index
        test_stats_df = test_stats_df.merge(self.detector_meta_df, how="inner", left_on="selected_model", right_on="model_idx")
        test_stats_df.drop('model_idx', axis=1, inplace=True)
        return test_stats_df, plot_df

class CVTesting(SplitTesting):
    def __init__(
        self,
        detection_mdl_dict,
        feature_names,
        test_axes,
        param_dict,
        cv: int,
        n_boot,
        tolerance_prob,
        alternative: str,
        random_state: int = 0
    ):
        """
        @param cv: number of CV splits
        all other params defined the same as SplitTesting
        """
        assert cv > 1
        self.cv = cv
        super(CVTesting, self).__init__(
            detection_mdl_dict,
            feature_names,
            test_axes,
            param_dict,
            cv/(cv + 1), # this value will be ignored
            n_boot,
            tolerance_prob,
            alternative,
            random_state=random_state,
        )

    def _test_one_sided(
        self, np_X, np_Y, ml_mdl, test_greater=True
    ):
        np.random.seed(self.random_state)
        pred_prob = ml_mdl.predict_proba(np_X)[:, 1]
        residuals = self._get_residual(np_Y, pred_prob, test_greater).reshape((1,-1))
        
        kf = KFold(n_splits=self.cv)
        all_pred_vals = []
        self.detectors = []
        for fold_idx, (train_idxs, test_idxs) in enumerate(kf.split(np_X)):
            detectors_fold, self.detector_meta_df = self._fit_detectors(
                np_X, np_Y, pred_prob, train_idxs, test_greater=test_greater
            )
            pred_val_list = self._get_pred_val_list(detectors_fold, self.detector_meta_df, np_X, pred_prob, test_idxs, test_greater)
            # Concatenate all predictions
            if len(all_pred_vals) == 0:
                self.detectors = [[detector] for detector in detectors_fold]
                all_pred_vals = pred_val_list
            else:
                for i, (pred_vals, detector) in enumerate(zip(pred_val_list, detectors_fold)):
                    self.detectors[i].append(detector)
                    all_pred_vals[i] = np.concatenate([all_pred_vals[i], pred_vals])
        
        test_stats_df, plot_df = self._get_test_stats(all_pred_vals, self.detector_meta_df, residuals, pred_prob=pred_prob, test_greater=test_greater, do_print=True)
        print("test_stats", test_stats_df)
        logging.info("test_stats %s", test_stats_df)

        # Simulate the test statistic under the worst-case null distribution
        boot_dfs = self._simulate_null_res(pred_prob, all_pred_vals, self.detector_meta_df, test_greater)
        boot_dfs = self._simulate_null_res(pred_prob, all_pred_vals, self.detector_meta_df)
        test_stats_df['pval'] = [np.mean(test_stats_df.test_stat[col_name] <= boot_dfs[col_name]) for col_name in test_stats_df.index]        
        test_stats_df['method'] = test_stats_df.index
        test_stats_df = test_stats_df.merge(self.detector_meta_df, how="inner", left_on="selected_model", right_on="model_idx")
        test_stats_df.drop('model_idx', axis=1, inplace=True)
        
        return test_stats_df, plot_df

class CVTestingTwoSided(SplitTesting):
    def __init__(
        self,
        detection_mdl_dict,
        feature_names,
        test_axes,
        param_dict,
        cv,
        n_boot,
        tolerance_prob,
        alternative: str,
        random_state: int = 0
    ):
        assert cv > 1
        self.cv = cv
        assert alternative == "both"
        super(CVTestingTwoSided, self).__init__(
            detection_mdl_dict,
            feature_names,
            test_axes,
            param_dict,
            cv/(cv + 1),
            n_boot,
            tolerance_prob,
            alternative,
            random_state=random_state,
        )

    def _test_two_sided(
        self, np_X, np_Y, ml_mdl
    ):
        logging.info("CV TESTING")
        np.random.seed(self.random_state)
        pred_prob = ml_mdl.predict_proba(np_X)[:, 1]
        residuals_greater = self._get_residual(np_Y, pred_prob, test_greater=True).reshape((1,-1))
        residuals_less = -self._get_residual(np_Y, pred_prob, test_greater=False).reshape((1,-1))
        
        kf = KFold(n_splits=self.cv)
        all_pred_vals = []
        self.detectors = []
        for fold_idx, (train_idxs, test_idxs) in enumerate(kf.split(np_X)):
            detectors_fold, self.detector_meta_df = self._fit_detectors(
                np_X, np_Y, pred_prob, train_idxs
            )
            pred_val_list = self._get_pred_val_list(detectors_fold, self.detector_meta_df, np_X, pred_prob, test_idxs)
            if len(all_pred_vals) == 0:
                self.detectors = [[detector] for detector in detectors_fold]
                all_pred_vals = pred_val_list
            else:
                for i, (pred_vals, detector) in enumerate(zip(pred_val_list, detectors_fold)):
                    self.detectors[i].append(detector)
                    all_pred_vals[i] = np.concatenate([all_pred_vals[i], pred_val_list[i]])
        
        test_stats_df, plot_df = self._get_test_stats_two_sided(all_pred_vals, self.detector_meta_df, residuals_greater, residuals_less, pred_prob=pred_prob, do_print=True)
        print("test_stats two sided", test_stats_df)
        logging.info("test_stats two sided%s ", test_stats_df)

        # Simulate the test statistic under the worst-case null distribution
        boot_dfs = self._simulate_null_res(pred_prob, all_pred_vals, self.detector_meta_df)
        test_stats_df['pval'] = [np.mean(test_stats_df.test_stat[col_name] <= boot_dfs[col_name]) for col_name in test_stats_df.index]        
        test_stats_df['method'] = test_stats_df.index
        test_stats_df = test_stats_df.merge(self.detector_meta_df, how="inner", left_on="selected_model", right_on="model_idx")
        test_stats_df.drop('model_idx', axis=1, inplace=True)
        
        return test_stats_df, plot_df


class SplitScore(SplitTesting):
    """
    Adaptive score-based CUSUM test, sample-splitting version
    """
    num_min = 4 # minimum number of samples to run a CUSUM test
    step_size = 20
    weight_min_idx = 10
    main_statistic = "cusum"
    mean_shift = 2

    def _get_g_standardized(self, pred_val, pred_prob):
        new_pred_prob = np.minimum(1 - 1e-10, np.maximum(1e-10, pred_prob + pred_val))
        g_standardized = pred_val/np.sqrt(new_pred_prob * (1 - new_pred_prob))
        if np.var(g_standardized) > 0:
            normalized_gval_std = g_standardized/np.sqrt(np.var(g_standardized))
            return np.maximum(0, normalized_gval_std - normalized_gval_std.mean() + self.mean_shift)
        else:
            return self._get_intercept(pred_val)
    
    def _get_g(self, pred_val, pred_prob=None):
        if np.var(pred_val) > 0:
            normalized_gval = pred_val/np.sqrt(np.var(pred_val))
            shifted_gval = normalized_gval - normalized_gval.mean() + self.mean_shift
            return np.maximum(0, shifted_gval)
        else:
            return self._get_intercept(pred_val)

    def _get_intercept(self, pred_val, pred_prob=None):
        return np.ones(pred_val.size) * self.mean_shift

    def _get_rank_standardized(self, pred_val, pred_prob):
        new_pred_prob = np.minimum(1 - 1e-10, np.maximum(1e-10, pred_prob + pred_val))
        g_standardized = pred_val/np.sqrt(new_pred_prob * (1 - new_pred_prob))
        sorted_idxs = np.argsort(g_standardized)
        seq = np.empty_like(pred_val, dtype=int)
        seq[sorted_idxs] = np.arange(pred_val.size)
        if np.var(seq) > 0:
            normalized_rank = seq/np.sqrt(np.var(seq))
            return np.maximum(0, normalized_rank - normalized_rank.mean() + self.mean_shift)
        else:
            return self._get_intercept(pred_val)
    
    def _get_rank(self, pred_val, pred_prob=None):
        seq = np.arange(pred_val.size, 0, -1)
        if np.var(seq) > 0:
            normalized_rank = seq/np.sqrt(np.var(seq))
            shifted_rank = normalized_rank - normalized_rank.mean() + self.mean_shift
            return np.maximum(0, shifted_rank)
        else:
            return self._get_intercept(pred_val)
    
    @property
    def z_func_dict(self):
        return {
            'gval': self._get_g,
            # 'gval_std': self._get_g_standardized,
            # 'rank_std': self._get_rank_standardized,
            'intercept': self._get_intercept,
            'rank': self._get_rank,
        }

    def _get_mask(self, pred_vals, min_num=1):
        """Which observations should we calculate the score with respect to.
        We remove the observations with predicted g(X) below 0

        Args:
            pred_vals (_type_): _description_
            min_num (int, optional): _description_. Defaults to 1.

        Returns:
            a binary mask for which observations to keep
        """
        num_mask = np.sum(pred_vals > 0)
        if num_mask >= min_num:
            return pred_vals > 0
        else:
            # If there are not enough elements above 0, use the i-th largest value of pred_vals
            partition = np.partition(pred_vals, pred_vals.size - min_num)
            thres = partition[pred_vals.size - min_num]
            return pred_vals > thres

    def _get_test_stats(self, pred_val_list, detector_meta_df, test_residuals, null_prob=None, pred_prob=None, test_greater=None, do_print=False) -> pd.Series:
        """
        Calculate the CUSUM statistic up to specific quantiles
        and the test statistic at specific quantiles.
        """
        test_stats_list = []
        cumsum_list = []
        for idx, pred_vals in enumerate(pred_val_list):
            sort_idxs = np.flip(np.argsort(pred_vals))
            all_pred_vals_sorted = pred_vals[sort_idxs]

            mask = self._get_mask(all_pred_vals_sorted)
            if do_print:
                print("mask", mask.sum())
                logging.info("mask %d", mask.sum())
            
            pred_vals_sorted = all_pred_vals_sorted[mask]
            test_residual_sorted = test_residuals[:,sort_idxs][:,mask]
            pred_prob_sorted = pred_prob[sort_idxs][mask]
            for z_func_name, z_func in self.z_func_dict.items():
                if mask.sum() < self.num_min:
                    if do_print:
                        logging.info("fill in %s", z_func_name)
                        print(z_func_name, "fill in", idx, mask.sum())

                    test_stats_list.append(pd.DataFrame({
                        "model": [idx] * 2 * test_residuals.shape[0],
                        "z_func": [z_func_name] * 2 * test_residuals.shape[0],
                        "agg": np.tile(["max", "last"], test_residuals.shape[0]),
                        "test_stat": [-np.inf] * 2 * test_residuals.shape[0]}))
                    continue

                z_vals_sorted = z_func(pred_vals_sorted, pred_prob_sorted)
                assert np.all(z_vals_sorted >= 0)
                test_score_sorted = test_residual_sorted * z_vals_sorted
                
                cumsums = np.cumsum(test_score_sorted, axis=1)/np.sqrt(test_score_sorted.shape[1])
                cusum = cumsums.max(axis=1)
                pred_pos_mean = cumsums[:,-1]
                
                if do_print:
                    cumsum_var = np.arange(1, 1 + test_score_sorted.shape[1])
                    cumsum_list.append(pd.DataFrame({
                        'cusum': cumsums.flatten(),
                        'idx': np.arange(mask.sum()),
                        'normalized_x': cumsum_var/cumsum_var[-1],
                        'residual_x': cumsum_var,
                    }))
                    cumsum_list[-1]["setting"] = "%s_0" % z_func_name
                    cumsum_list[-1]["model"] = detector_meta_df.name[idx]
                    cumsum_list[-1]["alternative"] = self.alternative
                    cumsum_list[-1]["name"] = self.__class__.__name__

                
                test_stats_list.append(pd.DataFrame({
                    "model": [idx] * cusum.shape[0] * 2,
                    "z_func": [z_func_name] * cusum.shape[0] * 2,
                    "agg": np.repeat(["max", "last"], cusum.shape[0]),
                    "test_stat": np.concatenate([cusum, pred_pos_mean], axis=0),
                    "replicate": np.tile(np.arange(cusum.shape[0]), 2)
                }))
            
        test_stats_df = pd.concat(test_stats_list).reset_index(drop=True)
        if do_print:
            logging.info("orig test stats %s", test_stats_df)
            print(test_stats_df)

        # find the best model
        test_stats_z_df = test_stats_df.iloc[test_stats_df.groupby(['replicate', 'agg', 'z_func']).test_stat.idxmax()].rename({'model': 'selected_model', 'z_func': 'selected_z_func'}, axis=1)
        test_stats_z_df['z_func'] = test_stats_z_df['selected_z_func']
        # find the best model and h-func
        test_stats_vec_df = test_stats_df.iloc[test_stats_df.groupby(['replicate', 'agg']).test_stat.idxmax()].rename({'model': 'selected_model', 'z_func': 'selected_z_func'}, axis=1)
        test_stats_vec_df['z_func'] = 'vec'
        test_stats_agg_df = pd.concat([test_stats_z_df, test_stats_vec_df])
        test_stats_agg_df.index = test_stats_agg_df['agg'] + "_" + test_stats_agg_df['z_func']
        test_stats_agg_df['method'] = test_stats_agg_df['agg'] + "_" + test_stats_agg_df['z_func']

        return test_stats_agg_df, pd.concat(cumsum_list) if len(cumsum_list) else None

    def plot(self, plot_df: pd.DataFrame, plot_file: str):
        """
        Creates control chart
        """
        if plot_df is not None:
            print(plot_df)
            plot_df['class_name'] = plot_df.model.str.split("_", expand=True)[0]
            plot_df = plot_df.rename({
                'class_name': 'Detector',
                'normalized_x': 'Quantile',
                'cusum': 'Cumulative score',
            }, axis=1)
            
            a = sns.relplot(
                data=plot_df[plot_df.setting == "gval_0"].reset_index(drop=True),
                x='Quantile',
                y='Cumulative score',
                hue='Detector',
                style="model",
                kind="line",
                legend=False)
            print(a._legend_data)
            cols = sns.color_palette()
            name_to_color = {
                'Random Forest': cols[0],
                'Kernel Logistic': cols[1],
            }
            patches = [matplotlib.lines.Line2D([0], [0], color=v, label=k) for k,v in
                    name_to_color.items()]

            plt.legend(handles=patches)
            plt.savefig(plot_file)
    
    def clean_for_saving(self, test_stat_df, cols_to_save=['max_gval']):
        """
        Run this code prior to pickling the entire detector.
        Saves only the best residual models
        (For CV models, we only store the residual models for the last fold)
        """
        best_idx_set = test_stat_df[test_stat_df.method == cols_to_save[0]].selected_model.unique()
        for idx in range(len(self.detectors)):
            if idx not in best_idx_set:
                self.detectors[idx] = None
    

class SplitScoreTwoSided(SplitScore, SplitTestingTwoSided):
    """
    Adaptive score-based CUSUM, two-sided, sample-splitting
    """
    main_statistic = "double_cusum"
    one_sided = False
    def _get_test_stats_two_sided(self,
                                  pred_val_list: List, 
                                  detector_meta_df: Dict,
                                  test_residuals_greater: np.ndarray,
                                  test_residuals_less: np.ndarray,
                                  null_prob=None, pred_prob=None, do_print: bool=False) -> pd.Series:
        """
        Calculate the CUSUM statistic up to specific quantiles
        and the test statistic at specific quantiles.

        @param test_residuals_greater: the residuals compared to the upper tolerance region (Y - p_delta(X))
        @param test_residuals_less: the residuals compared to the lower tolerance region (Y - p_-delta(X))
        """
        test_stats_list = []
        cumsum_list = []
        for idx, pred_vals in enumerate(pred_val_list):
            # repurposes the one-sided test statistic calculation, after we multiply the residuals with
            # their appropriate signs
            signed_test_residuals = (
                test_residuals_greater * (pred_vals > 0)
                - test_residuals_less * (pred_vals < 0)
            )
            test_stats_df, cumsums = self._get_test_stats(
                [np.abs(pred_vals)],
                detector_meta_df.iloc[[idx]].reset_index(),
                signed_test_residuals,
                null_prob=null_prob,
                pred_prob=pred_prob,
                do_print=do_print)
            test_stats_df["selected_model"] = idx
            test_stats_df = test_stats_df.rename({"selected_model": "model"}, axis=1).drop('method', axis=1)

            test_stats_list.append(test_stats_df)
            cumsum_list.append(cumsums)
        
        test_stats_df = pd.concat(test_stats_list)
        if do_print:
            print("test_stats_ALL_df", test_stats_df)
        test_stats_df = test_stats_df[test_stats_df.z_func != 'vec'].reset_index(drop=True)
        test_stats_df.drop('selected_z_func', axis=1, inplace=True)
        if do_print:
            logging.info("orig test stats %s", test_stats_df)
        
        test_stats_z_df = test_stats_df.iloc[test_stats_df.groupby(['replicate', 'agg', 'z_func']).test_stat.idxmax()]
        test_stats_z_df = test_stats_z_df.rename({'model': 'selected_model', 'z_func': 'selected_z_func'}, axis=1)
        test_stats_z_df['z_func'] = test_stats_z_df['selected_z_func']
        test_stats_vec_df = test_stats_df.iloc[test_stats_df.groupby(['replicate','agg']).test_stat.idxmax()].rename({'model': 'selected_model', 'z_func': 'selected_z_func'}, axis=1)
        test_stats_vec_df['z_func'] = 'vec'
        test_stats_agg_df = pd.concat([test_stats_z_df, test_stats_vec_df])
        test_stats_agg_df.index = test_stats_agg_df['agg'] + "_" + test_stats_agg_df['z_func']
        test_stats_agg_df['method'] = test_stats_agg_df['agg'] + "_" + test_stats_agg_df['z_func']
        test_stats_agg_df.index.name = "index"
        return test_stats_agg_df, pd.concat(cumsum_list) if any([c is not None for c in cumsum_list]) else None

class CVScore(CVTesting, SplitScore):
    """
    CV Adaptive score-based CUSUM test, one-sided
    """
    def __init__(
        self,
        detection_mdl,
        feature_names,
        test_axes,
        param_dict,
        cv,
        n_boot,
        tolerance_prob,
        alternative: str,
        random_state: int = 0
    ):
        assert cv > 1
        super(CVScore, self).__init__(
            detection_mdl,
            feature_names,
            test_axes,
            param_dict,
            cv,
            n_boot,
            tolerance_prob,
            alternative,
            random_state=random_state)

    def get_feat_imports(self, test_stats_df, orig_ml_mdl, test_X, test_Y, n_repeats: int = 1, col_names: str = ["max_gval"]) -> pd.DataFrame:
        """
        Calculate feature importance, relative to test statistic
        """
        assert self.alternative != "both"
        test_greater = self.alternative == "greater"
        
        pred_prob = orig_ml_mdl.predict_proba(test_X)[:,1]
        test_X_aug = self._get_x_features(test_X, pred_prob)
        
        residuals = self._get_residual(test_Y, pred_prob, test_greater=test_greater).reshape((1,-1))
        
        mdl_pred_logit = np.log(pred_prob/(1 - pred_prob))

        test_sign = 1 if test_greater else -1
        null_prob = make_prob(pred_prob + self.tolerance_prob * test_sign)
        
        kf = KFold(n_splits=self.cv)
        all_feat_imports = []
        for col_name in col_names:
            print("test_stats_df", test_stats_df)
            selected_model_idx = test_stats_df.selected_model[test_stats_df.method == col_name].iloc[0]
            print("selected_model_idx", selected_model_idx)
            print("self.detector_meta_df", self.detector_meta_df)
            selected_detectors = self.detectors[selected_model_idx]
            detector_meta_dict = self.detector_meta_df.iloc[selected_model_idx]

            def mdl_scorer(residual_mdls, X_feats, residuals):
                # Aggregate predictions across all folds
                all_pred_vals = None
                for fold_idx, (train_idxs, test_idxs) in enumerate(kf.split(X_feats)):
                    if detector_meta_dict['class_name'] in CLASSIFIERS:
                        # We don't want to shuffle the input to the ML model
                        # Only want to shuffle what goes into the detector...
                        # so we have to swap out the prediction from the ML model for the shuffled input
                        # and add back in the ML prediction for the original input
                        shuff_pred_prob = residual_mdls[fold_idx].predict_proba(X_feats[test_idxs])[:,1]
                        shuff_pred_logit = to_safe_logit(shuff_pred_prob)
                        pred_logit_test = shuff_pred_logit - X_feats[test_idxs,0] + mdl_pred_logit[test_idxs]
                        pred_prob_test = 1/(1 + np.exp(-pred_logit_test))
                        pred_val_list = (pred_prob_test - null_prob[test_idxs]) * test_sign
                    elif detector_meta_dict['class_name'] in REGRESSORS:
                        pred_val_list = self._get_pred_vals(residual_mdls[fold_idx], X_feats[test_idxs], pred_prob[test_idxs], detector_meta_dict.class_name[selected_model_idx], test_greater=test_greater)
                    else:
                        raise ValueError("detector name not recognized")
                    
                    if all_pred_vals is None:
                        all_pred_vals = pred_val_list
                    else:
                        all_pred_vals = np.concatenate([all_pred_vals, pred_val_list])

                test_stat_df, _ = self._get_test_stats(
                    [all_pred_vals],
                    [detector_meta_dict],
                    residuals,
                    pred_prob=pred_prob,
                    test_greater=test_greater
                )
                return test_stat_df.test_stat.to_dict()

            feat_res_dict = permutation_importance(selected_detectors, test_X_aug, residuals, scoring=mdl_scorer, n_repeats=n_repeats)
            for test_stat_name, feat_res in feat_res_dict.items():
                feat_import = pd.DataFrame({
                    "feature": np.concatenate([["orig_pred"], self.feature_names[self.axes]]),
                    "importance": feat_res['importances_mean']}).sort_values(by="importance", ascending=False)
                feat_import['test_stat'] = test_stat_name
                feat_import['residual_model'] = col_name
                print(feat_import)
                logging.info(feat_import)
                all_feat_imports.append(feat_import)
        all_feat_imports = pd.concat(all_feat_imports).reset_index(drop=True)
        return all_feat_imports


class CVScoreTwoSided(CVTestingTwoSided, SplitScoreTwoSided):
    """
    Adaptive score-based CUSUM, two-sided, CV
    """
    one_sided = False
    def __init__(self,
        detection_mdl,
        feature_names,
        test_axes,
        param_dict,
        cv,
        n_boot,
        tolerance_prob,
        alternative: str,
        random_state: int = 0,
    ):
        self.cv = cv
        assert alternative == "both"
        super(CVScoreTwoSided, self).__init__(
            detection_mdl,
            feature_names,
            test_axes,
            param_dict,
            cv,
            n_boot,
            tolerance_prob,
            alternative,
            random_state=random_state,
        )

    def get_feat_imports(self, test_stats_df, orig_ml_mdl, test_X, test_Y, n_repeats: int = 1, col_names: str = ["max_gval"]) -> pd.DataFrame:
        """
        Calculate feature importance, relative to test statistic
        """
        pred_prob = orig_ml_mdl.predict_proba(test_X)[:,1]
        test_X_aug = self._get_x_features(test_X, pred_prob)
        
        residuals_greater = self._get_residual(test_Y, pred_prob, test_greater=True).reshape((-1,1))
        residuals_less = -self._get_residual(test_Y, pred_prob, test_greater=False).reshape((-1,1))
        residuals = np.concatenate([residuals_greater, residuals_less], axis=1)
        
        mdl_pred_logit = np.log(pred_prob/(1 - pred_prob))

        null_prob_greater = make_prob(pred_prob + self.tolerance_prob)
        null_prob_less = make_prob(pred_prob - self.tolerance_prob)

        kf = KFold(n_splits=self.cv)
        all_feat_imports = []
        for col_name in col_names:
            selected_model_idx = test_stats_df.selected_model[test_stats_df.method == col_name].to_numpy()[0]
            print(selected_model_idx)
            selected_detectors = self.detectors[selected_model_idx]
            detector_meta_dict = self.detector_meta_df.iloc[selected_model_idx]

            def mdl_scorer(residual_mdls, X_feats, residuals):
                # Aggregate predictions across all folds
                all_pred_vals = None
                for fold_idx, (train_idxs, test_idxs) in enumerate(kf.split(X_feats)):
                    if detector_meta_dict.class_name in CLASSIFIERS:
                        # We don't want to shuffle the input to the ML model
                        # Only want to shuffle what goes into the detector...
                        # so we have to swap out the prediction from the ML model for the shuffled input
                        # and add back in the ML prediction for the original input
                        shuff_pred_prob = residual_mdls[fold_idx][0].predict_proba(X_feats[test_idxs])[:,1]
                        shuff_pred_logit = to_safe_logit(shuff_pred_prob)
                        pred_logit_test = shuff_pred_logit - X_feats[test_idxs,0] + mdl_pred_logit[test_idxs]
                        pred_prob_test = 1/(1 + np.exp(-pred_logit_test))
                        pred_val_greater = pred_prob_test - null_prob_greater[test_idxs]
                        pred_val_less = pred_prob_test - null_prob_less[test_idxs]
                        pred_val_list = self._combine_pred_vals(pred_val_greater, pred_val_less)
                    elif detector_meta_dict.class_name in REGRESSORS:
                        pred_val_list = self._get_pred_vals(residual_mdls[fold_idx], X_feats[test_idxs], pred_prob[test_idxs], detector_meta_dict.class_name)
                    else:
                        raise ValueError("detector name not recognized")
                    
                    if all_pred_vals is None:
                        all_pred_vals = pred_val_list
                    else:
                        all_pred_vals = np.concatenate([all_pred_vals, pred_val_list])
                perm_test_stat, _ = self._get_test_stats_two_sided(
                    [all_pred_vals],
                    self.detector_meta_df.iloc[[selected_model_idx]],
                    residuals[:,0].reshape((1,-1)),
                    residuals[:,1].reshape((1,-1)),
                    pred_prob=pred_prob
                )
                return perm_test_stat.test_stat.to_dict()

            feat_res_dict = permutation_importance(selected_detectors, test_X_aug, residuals, scoring=mdl_scorer, n_repeats=n_repeats)
            for test_stat_name, feat_res in feat_res_dict.items():
                feat_import = pd.DataFrame({
                    "feature": np.concatenate([["orig_pred"], self.feature_names[self.axes]]),
                    "importance": feat_res['importances_mean']}).sort_values(by="importance", ascending=False)
                feat_import['test_stat'] = test_stat_name
                feat_import['residual_model'] = col_name
                print(feat_import)
                logging.info(feat_import)
                all_feat_imports.append(feat_import)
        all_feat_imports = pd.concat(all_feat_imports).reset_index(drop=True)
        return all_feat_imports


class LogisticRecalibration(SplitScore):
    main_statistic = "platt"
    one_sided = True
    def __init__(
        self,
        n_boot,
        tolerance_prob,
        alternative,
        random_state: int = 0,
    ):
        self.axes = [0]
        self.split = 0
        self.n_boot = n_boot
        self.tolerance_prob = tolerance_prob
        self.alternative = alternative
        self.random_state = random_state
    
    def _fit_detectors(
        self, np_X, np_Y, pred_prob, train_idxs, test_greater=True,
    ):
        """
        This doesnt actually fit anything. This returns a single detector and it simply
        returns the predicted probability from the original ML algorithm
        """
        detector = LinearRegression(fit_intercept=False)
        np_X_aug = self._get_x_features(np_X, pred_prob)
        # This is just to initialize things. We override it in the next line
        detector.fit(np_X_aug, np_Y)
        # just need some large positive number so logistic recalibration procedure will have a non-negative valued detector
        detector.intercept_ = 10
        detector.coef_[:] = 0
        detector.coef_[0] = 1

        model_meta_df = pd.DataFrame({
            'model_idx': [0],
            'class_name': 'LinearRegression',
            'name': 'LinearRegression_0'
            })
        return [detector], model_meta_df

class LogisticRecalibrationTwoSided(SplitScoreTwoSided):
    main_statistic = "platt"
    one_sided = False
    def __init__(
        self,
        n_boot,
        tolerance_prob,
        alternative,
        random_state: int = 0,
    ):
        self.axes = [0]
        self.split = 0
        self.n_boot = n_boot
        self.tolerance_prob = tolerance_prob
        self.alternative = alternative
        self.random_state = random_state
    
    def _fit_detectors(
        self, np_X, np_Y, pred_prob, train_idxs,
    ):
        """
        Doesn't actually fit a model. Only returns a model that returns the predicted probability
        from the original ML model
        """
        detector = LinearRegression(fit_intercept=False)
        np_X_aug = self._get_x_features(np_X, pred_prob)
        # This is just to initialize things. We override it in the next line
        detector.fit(np_X_aug, np_Y)
        print(detector.coef_)
        # just need some large positive number so logistic recalibration procedure will have a non-negative valued detector
        detector.intercept_ = 10
        detector.coef_[:] = 0
        detector.coef_[0] = 1

        model_meta_df = pd.DataFrame({
            'model_idx': [0],
            'class_name': 'LinearRegression',
            'name': 'LinearRegression_0'
            })
        return [[detector, detector]], model_meta_df


class SplitChiSquared(SplitTesting):
    """
    Splits the data first to learn residuals
    Performs H-L test in the remaining data with squared O-E residuals, with n_splits, sorted along the learned axis
    similar to Zhang 2021
    """
    main_statistic = "adaptchisq"
    n_bin_list = [2,4,8,10]

    def _get_mask(self, pred_vals):
        return np.ones(pred_vals.shape, dtype=bool)

    def _get_test_stats(self, pred_val_list, detector_meta_df, test_residual, null_prob=None, pred_prob=None,
            test_greater=None, do_print=False):
        """
        Calculate the chi-sq statistic summed across bins
        """
        test_stats_list = []
        for idx, pred_vals in enumerate(pred_val_list):
            sort_option = np.flip(np.argsort(pred_vals))
            test_residual_sorted = test_residual[:,sort_option]
            pred_prob_sorted = pred_prob[sort_option]
            hl_stats = []
            for n_bins in self.n_bin_list:
                split_idxs = np.array_split(np.arange(test_residual_sorted.shape[1]), n_bins)
                grp_vars = np.array([
                    np.sum(pred_prob_sorted[grp_idxs] * (1 - pred_prob_sorted[grp_idxs]))
                    for grp_idxs in split_idxs])
                if np.any(grp_vars == 0):
                    continue
                hl_cells = np.array([
                    np.sum(test_residual_sorted[:,grp_idxs], axis=1)
                    for grp_idxs in split_idxs]).T
                hl_normalized_cells = np.power(np.maximum(hl_cells, 0), 2)/grp_vars
                if do_print:
                    print("grp_vars", grp_vars)
                    print("hl_cells", hl_cells)
                    print("hl_normalized_cells", hl_normalized_cells)
                    logging.info("hl_normalized_cells %s", hl_normalized_cells)
                hl_stats.append(np.sum(hl_normalized_cells, axis=1))
            test_stats_list.append(pd.DataFrame({
                "model": [idx] * len(self.n_bin_list) * test_residual.shape[0],
                "test_stat": np.concatenate(hl_stats, axis=0),
                "n_bin": np.repeat(self.n_bin_list, test_residual.shape[0]),
                "replicate": np.tile(np.arange(test_residual.shape[0]), len(self.n_bin_list))
            }))
            print(test_stats_list[-1])
        test_stats_df = pd.concat(test_stats_list).reset_index(drop=True)
        
        test_stats_df = test_stats_df.iloc[test_stats_df.groupby(['replicate','n_bin']).test_stat.idxmax()].rename({'model': 'selected_model'}, axis=1)
        test_stats_df.index = "adaptchisq_" + (test_stats_df["n_bin"].astype(str))
        test_stats_df['method'] = "adaptchisq_" + (test_stats_df["n_bin"].astype(str))
        test_stats_df.index.name = "index"
        
        return test_stats_df, None

class SplitChiSquaredTwoSided(SplitTestingTwoSided, SplitChiSquared):
    """
    Splits the data first to learn residuals
    Performs H-L test in the remaining data with squared O-E residuals, with n_splits, sorted along the learned axis
    similar to Zhang 2021
    """
    main_statistic = "adaptchisq"

    def _get_test_stats_two_sided(self, pred_val_list, detector_meta_df, test_residuals_greater, test_residuals_less, null_prob=None, pred_prob=None,
            test_greater=None, do_print=False):
        """
        Calculate the chi-sq statistic summed across bins
        """
        test_stats_list = []
        for idx, pred_vals in enumerate(pred_val_list):
            signed_test_residuals = (
                test_residuals_greater * (pred_vals > 0)
                - test_residuals_less * (pred_vals < 0)
            )
            test_stats_df, _ = self._get_test_stats(
                [np.abs(pred_vals)],
                detector_meta_df.iloc[[idx]],
                signed_test_residuals,
                null_prob=null_prob,
                pred_prob=pred_prob,
                do_print=do_print)
            test_stats_df["selected_model"] = idx
            test_stats_df = test_stats_df.rename({"selected_model": "model"}, axis=1).drop('method', axis=1)
            test_stats_list.append(test_stats_df)
            
        test_stats_df = pd.concat(test_stats_list).reset_index(drop=True)
        logging.info('chitest %s', test_stats_df)
        
        test_stats_df = test_stats_df.iloc[test_stats_df.groupby(['replicate', 'n_bin']).test_stat.idxmax()]
        test_stats_df = test_stats_df.rename({'model': 'selected_model'}, axis=1)
        test_stats_df.index = "adaptchisq_" + (test_stats_df["n_bin"].astype(str))
        test_stats_df['method'] = "adaptchisq_" + (test_stats_df["n_bin"].astype(str))
        test_stats_df.index.name = "index"
        return test_stats_df, None

class CVChiSquared(CVTesting, SplitChiSquared):
    main_statistic = "CVchisq"
    def __init__(
        self,
        detection_mdl,
        feature_names,
        axes,
        param_dict,
        cv,
        n_boot,
        tolerance_prob,
        alternative,
        random_state: int = 0
    ):
        self.cv = cv
        super(CVChiSquared, self).__init__(
            detection_mdl,
            feature_names,
            axes,
            param_dict,
            cv,
            n_boot,
            tolerance_prob,
            alternative,
            random_state=random_state,
        )

class CVChiSquaredTwoSided(CVTestingTwoSided, SplitChiSquaredTwoSided):
    main_statistic = "CVchisq2"
    def __init__(
        self,
        detection_mdl,
        feature_names,
        axes,
        param_dict,
        cv,
        n_boot,
        tolerance_prob,
        alternative,
        random_state: int = 0
    ):
        self.cv = cv
        super(CVChiSquaredTwoSided, self).__init__(
            detection_mdl,
            feature_names,
            axes,
            param_dict,
            cv,
            n_boot,
            tolerance_prob,
            alternative,
            random_state=random_state,
        )
