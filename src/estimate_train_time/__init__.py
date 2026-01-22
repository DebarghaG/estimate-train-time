"""
estimate-train-time: Distributed training time estimator for Large Language Models

This package provides tools to estimate the time required to train LLMs
across multiple GPUs using various parallelism strategies.
"""

__version__ = "0.1.0"

from estimate_train_time.estimator.prediction import one_batch_predict
from estimate_train_time.estimator.predictor import Predictor

__all__ = [
    "__version__",
    "one_batch_predict",
    "Predictor",
]
