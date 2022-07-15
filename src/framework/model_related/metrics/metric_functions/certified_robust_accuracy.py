import tensorflow as tf

from framework.general_code.directory import Directory
from abc import ABC

from framework.model_related.metrics.metric import Metric
from framework.model_related.metrics.metric_options import MetricOption


multipliers = {
    "L1": 1.,
    "L2": 2. ** (1 / 2),
    "Linf": 2.,
}


class GeneralCertifiedRobustAccuracy(Metric, ABC):
    """
    Define score_margin := score_of_correct_class - highest_other_score.
    We only have certified robustness if any possible change to the scores keeps the score_margin positive.
    Now write a and b for the change in those scores.
    We want that |a| + |b| < score_margin.
    We have that
    |a| + |b| <= 2 sqrt((a^2+b^2)/2) <= sqrt(2) ||score_change||_2,
    |a| + |b| <= 2 ||score_change||_{infty}, and
    |a| + |b| <= ||score_change||_1.
    Therefore, knowing that the (L1, L2 or Linf) norm of the score changes is bounded,
    gives us a way to guarantee |a| + |b| < score_margin in some cases.
    More precisely, we can guarantee certified robustness,
    if multiplier * maximal_change < score_margin,
    where multiplier depends on the kind of norm
    (L1=>1, L2=>sqrt(2), Linf=>2),
    and the maximal_change is with respect to that norm.
    """
    name = "cra"  # "CertifiedRobustAccuracy"

    def __init__(self, maximal_perturbation, norm="L2"):
        self.name = f"{self.name}{maximal_perturbation:.3g}"
        # self.threshold = threshold  # Maximal allowed change to the scores in self.norm-norm.
        self.maximal_perturbation = maximal_perturbation  # Maximal allowed change to the scores in self.norm-norm.
        if norm not in multipliers.keys():
            raise NotImplementedError(f"Norm \"{norm}\" not in implemented norms: {multipliers.keys()}")
        self.multiplier = multipliers[norm]
        super().__init__()


class CertifiedRobustAccuracyFromScore(GeneralCertifiedRobustAccuracy):
    def __call__(self, y_true_batch, s_pred_batch):
        y_true = tf.math.argmax(y_true_batch, axis=-1)
        s_pred_penalized = s_pred_batch - self.multiplier * self.maximal_perturbation * y_true_batch
        y_pred = tf.math.argmax(s_pred_penalized, axis=-1)
        return tf.cast(tf.equal(y_true, y_pred), tf.float32)


def register_cra_metrics(metric_directory: Directory):
    metric_directory.register(
        CertifiedRobustAccuracyFromScore,
        "certified_robust_accuracy",
        MetricOption.FROM_SCORES.value,
    )
