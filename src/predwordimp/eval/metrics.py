import numpy as np
from scipy.stats import kendalltau, pearsonr, somersd

from predwordimp.eval.util import get_rank_limit
from predwordimp.util.logger import get_logger

ranking_type = list[int] | np.ndarray
rankings = list[list[int]] | list[np.ndarray]
logger = get_logger(__name__)


class RankingEvaluator:
    @staticmethod
    def ignore_maximal(x: rankings, rank_limit: float | int = 0.0) -> rankings:
        """
        Modifies the rankings by ignoring the maximal value in each ranking and changing it to limit for number
        of selected-ranked words.

        If some user selected fewer words, the average rank will be smaller than rank_limit.

        Parameters:
            x (rankings): The list of rankings.
            to_limit_ranked (bool, optional): If True, limits the number of selected-ranked values based on the
                'ranked_limit' parameter. Default is False.
            ranked_limit (float | int): The threshold for limiting the number of selected-ranked values.

        Returns:
            rankings: The modified rankings after ignoring the maximal value and optionally limiting the number of
                selected-ranked values.
        """
        x_transformed = []
        for row in x:
            limit = get_rank_limit(rank_limit, len(row)) + 1
            maximal = max(row)
            row_transformed = [
                limit if x >= limit or x == maximal else x for x in row
            ]  # TODO decide rank for unselected words

            x_transformed.append(row_transformed)
        return x_transformed

    @staticmethod
    def get_selected_only(x: ranking_type) -> set[int]:
        return {i for i, val in enumerate(x) if val != max(x)}

    @staticmethod
    def get_selected_order(x: ranking_type) -> list[int]:
        selected = [(i, val) for i, val in enumerate(x) if val != max(x)]
        order = sorted(selected, key=lambda x: x[1])
        return [x[0] for x in order]

    @staticmethod
    def avg_overlap(x: list[int], y: list[int], depth: int) -> np.floating:
        overlaps = []

        for i in range(1, depth + 1):
            x_prefix = set(x[:i])
            y_prefix = set(y[:i])
            intersection = x_prefix.intersection(y_prefix)
            union = x_prefix.union(y_prefix)
            overlap = len(intersection) / i

            logger.debug(f"prefixes : {x_prefix, y_prefix}")
            logger.debug(f"inter: {intersection}")
            logger.debug(f"union : {union}")
            logger.debug(f"overlap : {overlap}")

            overlaps.append(overlap)

        return np.mean(overlaps)

    @staticmethod
    def avg_overlaps(preds, labels, limit: int | float) -> np.floating:
        RankingEvaluator.same_lengths(preds, labels)

        overlaps = []

        for pred, label in zip(preds, labels):
            RankingEvaluator.same_lengths(pred, label)
            depth = get_rank_limit(limit, len(pred))

            pred_selected = RankingEvaluator.get_selected_order(pred)
            label_selected = RankingEvaluator.get_selected_order(label)

            overlap = RankingEvaluator.avg_overlap(pred_selected, label_selected, depth)

            logger.debug(f"pred_selected : {pred_selected}")
            logger.debug(f"label_selected : {label_selected}")
            logger.debug(f"avg_overlap : {overlap}")
            logger.debug("===============")

            overlaps.append(overlap)

        return np.mean(overlaps)

    @staticmethod
    def same_lengths(x: rankings | ranking_type, y: rankings | ranking_type) -> None:
        if len(x) != len(y):
            raise ValueError(
                f"The lengths of preds and labels must be the same: {len(x)} != {len(y)}"
            )

    @staticmethod
    def remove_maximal_pairs(
        preds: rankings, labels: rankings
    ) -> tuple[rankings, rankings]:
        # RankingEvaluator.same_lengths(preds, labels)

        new_preds = []
        new_labels = []

        for pred, label in zip(preds, labels):
            last_rank = max(pred)

            if max(label) != last_rank:
                raise ValueError("Maximum ranks in preds and labels are not equal.")

            good_pairs = [
                (p, l)
                for p, l in zip(pred, label)
                if not ((p == l) and (p == last_rank))
            ]

            # good_pairs = [(p, l) for p, l in zip(pred, label) if p != last_rank and l != last_rank]

            new_preds.append([pair[0] for pair in good_pairs])
            new_labels.append([pair[1] for pair in good_pairs])

        return new_preds, new_labels

    @staticmethod
    def mean_rank_correlation(
        preds: rankings, labels: rankings, method: str, concatenate_lists: bool = True
    ) -> tuple[np.floating, list[float]]:
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
        p_values = []

        for i in range(num_samples):
            pred_ranking = preds[i]
            label_ranking = labels[i]

            RankingEvaluator.same_lengths(pred_ranking, label_ranking)

            if method == "pearson":
                # r = spearmanr(pred_ranking, label_ranking)
                corr, pval = pearsonr(pred_ranking, label_ranking)
                correlations.append(corr)
                p_values.append(pval)
            elif method == "kendall":
                tau, pval = kendalltau(pred_ranking, label_ranking)
                correlations.append(tau)
                p_values.append(pval)
            elif method == "somers":
                d = somersd(pred_ranking, label_ranking).statistic
                correlations.append(d)
            else:
                raise ValueError(
                    f"Invalid method: {method}. Choose from 'pearson', 'kendall', 'somers'."
                )

        mean_correlation = np.mean(correlations)
        logger.info(f"{method} min={min(correlations)}, max={max(correlations)}")
        return mean_correlation, p_values

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
