import numpy as np
from scipy.stats import kendalltau, somersd, spearmanr

from predwordimp.util.logger import get_logger

rankings = list[list[int]] | list[np.ndarray]
logger = get_logger(__name__)


class RankingEvaluator:
    # @staticmethod
    # def ignore_maximal(x: rankings) -> rankings:
    #     """
    #     Positions with the maximum rank value are for words that were not selected by any user. If some
    #     user selected fewer words, the average rank will be smaller than rank_limit, so predictions from the
    #     model are getting penalized for it. Give these words rank equal to sequence length.
    #     """
    #     x_transformed = []
    #     for row in x:
    #         max_value = max(row)
    #         x_transformed.append(
    #             [len(row) if x == max_value else x for x in row]
    #         )
    #     return x_transformed

    @staticmethod
    def mean_rank_correlation(
        preds: rankings, labels: rankings, method: str
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
        if len(preds) != len(labels):
            raise ValueError(
                f"The lengths of preds and labels must be the same: {len(preds)} != {len(labels)}"
            )

        num_samples = len(preds)
        correlations = []

        for i in range(num_samples):
            pred_ranking = preds[i]
            label_ranking = labels[i]

            if len(pred_ranking) != len(label_ranking):
                raise ValueError(
                    f"The lengths of predicted and ground truth rankings must be the same for sample {i}"
                )

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
