import numpy as np
from scipy.stats import kendalltau, somersd, spearmanr

from predwordimp.eval.util import get_rank_limit
from predwordimp.util.logger import get_logger

ranking_type = list[int] | np.ndarray
rankings = list[list[int]] | list[np.ndarray]
logger = get_logger(__name__)


class RankingEvaluator:
    @staticmethod
    def ignore_maximal(
        x: rankings, to_limit_ranked: bool = False, ranked_limit: float | int = 0.0
    ) -> rankings:
        """
        Positions with the maximum rank value are for words that were not selected by any user. If some
        user selected fewer words, the average rank will be smaller than rank_limit, so predictions from the
        model are getting penalized for it. Give these words rank equal to sequence length.
        """
        x_transformed = []
        for row in x:
            max_value = max(row)

            row_transformed = [len(row) if x == max_value else x for x in row]

            if to_limit_ranked:
                # row_transformed = RankingEvaluator.ranked_limit(row_transformed, ranked_limit, len(row) - 1)
                row_transformed = RankingEvaluator.ranked_limit(
                    row_transformed, ranked_limit, len(row)
                )

            x_transformed.append(row_transformed)
        return x_transformed

    @staticmethod
    def ranked_limit(x: ranking_type, r_limit: int | float, max_r: int) -> list[int]:
        limit = get_rank_limit(r_limit, len(x))
        return [r if r < limit else max_r for r in x]

    @staticmethod
    def get_selected_only(x: ranking_type) -> set[int]:
        return {i for i, val in enumerate(x) if val != max(x)}

    @staticmethod
    def same_lengths(x: rankings | ranking_type, y: rankings | ranking_type) -> None:
        if len(x) != len(y):
            raise ValueError(
                f"The lengths of preds and labels must be the same: {len(x)} != {len(y)}"
            )

    @staticmethod
    def mean_rank_correlation(
        preds: rankings, labels: rankings, method: str, concatenate_lists: bool = True
    ) -> np.floating:
        """
        Compute the mean rank correlation coefficient between predicted rankings and ground truth rankings.

        Args:
            preds (list[np.ndarray]): List of predicted rankings.
            labels (list[np.ndarray]): List of ground truth rankings.
            method (str): Method for computing rank correlation coefficient.
                Options: 'spearman', 'kendall', 'somers'.

        Returns:
            np.floating: Mean rank correlation coefficient.
        """
        RankingEvaluator.same_lengths(preds, labels)

        if concatenate_lists:
            preds = [[item for sublist in preds for item in sublist]]
            labels = [[item for sublist in labels for item in sublist]]

        num_samples = len(preds)
        correlations = []

        for i in range(num_samples):
            pred_ranking = preds[i]
            label_ranking = labels[i]

            RankingEvaluator.same_lengths(pred_ranking, label_ranking)

            if method == "spearman":
                rho, _ = spearmanr(pred_ranking, label_ranking)
                correlations.append(rho)
            elif method == "kendall":
                tau, _ = kendalltau(pred_ranking, label_ranking)
                correlations.append(tau)
            elif method == "somers":
                d = somersd(pred_ranking, label_ranking).statistic
                correlations.append(d)
            else:
                raise ValueError(
                    f"Invalid method: {method}. Choose from 'spearman', 'kendall', 'somers'."
                )

        mean_correlation = np.mean(correlations)
        logger.info(f"{method} min={min(correlations)}, max={max(correlations)}")
        return mean_correlation

    @staticmethod
    def has_least_intersection(pred: ranking_type, label: ranking_type, k: int) -> bool:
        pred_selected = RankingEvaluator.get_selected_only(pred)
        label_selected = RankingEvaluator.get_selected_only(label)
        return len(pred_selected.intersection(label_selected)) >= k

    @staticmethod
    def least_intersection(preds: rankings, labels: rankings, k: int):
        RankingEvaluator.same_lengths(preds, labels)

        hits = 0

        for pred, label in zip(preds, labels):
            RankingEvaluator.same_lengths(pred, label)
            if RankingEvaluator.has_least_intersection(pred, label, k):
                hits += 1

        return hits / len(preds)
