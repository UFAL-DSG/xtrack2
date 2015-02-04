import numpy as np
import spearmint


def main(job_id, params):
    print params
    return float(5 + 10 * np.random.randn(1))
