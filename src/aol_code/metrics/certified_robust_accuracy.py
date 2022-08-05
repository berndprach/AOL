import tensorflow as tf


multipliers = {
    "L1": 1.,
    "L2": 2. ** (1 / 2),
    "Linf": 2.,
}


class CertifiedRobustAccuracyFromScore:
    """
    Suppose we have a vector of scores s.
    We want to be sure that the predicted class from scores s+v
    (the argmax of the vector) is the correct class for all vectors
    v with bounded (L1, L2 or Linf) norm.

    The worst possible v has all but two entries equal to 0,
    call this entries a and b.

    An input is certifieably robustly classified if
    |a| + |b| < score_of_correct_class - highest_other_score.

    There is an upper bound of |a|+|b| in terms of L1, L2 and Linf norm of v:
    |a| + |b| <= 2 sqrt((a^2+b^2)/2) <= sqrt(2) || v ||_2,
    |a| + |b| <= 2 || v ||_{inf}, and
    |a| + |b| <= || v ||_1.

    Therefore, we have certified robust accuracy if the score margin
    is bigger than the maximal allowed perturbation of the score vector,
    multiplied with some (norm-dependent) multiplier.
    (1 for L1, sqrt(2) for L2, and 2 for Linf.)
    """
    basic_name = "cra"  # "CertifiedRobustAccuracy"

    def __init__(self, maximal_perturbation, norm="L2"):
        self.name = f"{self.basic_name}{maximal_perturbation:.3g}"
        self.maximal_perturbation = maximal_perturbation
        # Maximal allowed change to the scores measured by the given norm.
        if norm not in multipliers.keys():
            raise NotImplementedError(f"Norm \"{norm}\" not implemented!"
                                      f"Options: {multipliers.keys()}")
        self.multiplier = multipliers[norm]
        super().__init__()

    def __call__(self, y_true_batch, s_pred_batch):
        y_true = tf.math.argmax(y_true_batch, axis=-1)
        threshold = self.multiplier * self.maximal_perturbation
        threshold_vector = threshold * y_true_batch
        s_pred_penalized = s_pred_batch - threshold_vector
        y_pred = tf.math.argmax(s_pred_penalized, axis=-1)
        return tf.cast(tf.equal(y_true, y_pred), tf.float32)
