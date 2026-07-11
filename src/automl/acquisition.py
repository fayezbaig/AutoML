import numpy as np
from scipy.stats import norm


class ExpectedImprovement:

    def __init__(self):

        pass

    def compute(self, mean, std, best):

        if std < 1e-8:
            return 0

        improvement = mean - best

        z = improvement / std

        ei = improvement * norm.cdf(z) + std * norm.pdf(z)

        return ei