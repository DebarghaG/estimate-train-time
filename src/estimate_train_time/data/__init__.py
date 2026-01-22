"""
Data subpackage containing bundled regressor models and example configurations.
"""

import importlib.resources
import sys

if sys.version_info >= (3, 9):
    def get_data_path():
        """Get the path to the data directory."""
        return importlib.resources.files("estimate_train_time.data")

    def get_regressors_path():
        """Get the path to the regressors directory."""
        return importlib.resources.files("estimate_train_time.data") / "regressors"

    def get_examples_path():
        """Get the path to the examples directory."""
        return importlib.resources.files("estimate_train_time.data") / "examples"
else:
    import pkg_resources

    def get_data_path():
        """Get the path to the data directory."""
        return pkg_resources.resource_filename("estimate_train_time", "data")

    def get_regressors_path():
        """Get the path to the regressors directory."""
        return pkg_resources.resource_filename("estimate_train_time.data", "regressors")

    def get_examples_path():
        """Get the path to the examples directory."""
        return pkg_resources.resource_filename("estimate_train_time.data", "examples")

__all__ = ["get_data_path", "get_regressors_path", "get_examples_path"]
