# ABOUTME: Computes single-output forest energy means and uncertainties from the same terminal nodes.
# ABOUTME: Preserves the pinned serial estimator's seed draw, fitted trees and accumulation order.
import numpy as np
import sklearn
import skopt
from joblib import effective_n_jobs
from sklearn.utils import check_random_state
from skopt.learning.forest import ExtraTreesRegressor
from skopt.utils import cook_estimator


class EnergyForest(ExtraTreesRegressor):
    def predict(self, X, return_std=False):
        if (not return_std or self.criterion != "squared_error" or getattr(self, "n_outputs_", None) != 1
                or self.n_jobs != 1 or effective_n_jobs(1) != 1 or type(X) is not np.ndarray
                or X.dtype != np.float64 or X.ndim != 2 or not np.isfinite(X).all()):
            return super().predict(X, return_std=return_std)
        grid = self._validate_X_predict(X)
        mean = np.zeros(len(grid))
        second_moment = np.zeros(len(grid))
        for tree in self.estimators_:
            leaves = tree.apply(grid, check_input=False)
            value = tree.tree_.value[leaves, 0, 0]
            variance = tree.tree_.impurity[leaves]
            variance[variance < self.min_variance] = self.min_variance
            mean += value
            second_moment += variance + value**2
        mean /= len(self.estimators_)
        second_moment /= len(self.estimators_)
        second_moment -= mean**2.0
        second_moment[second_moment < 0.] = 0.
        return mean, second_moment**0.5


def forest_search_options(dimensions, seed):
    """Construct scikit-optimize's serial ET estimator and preserve the search RNG after its seed draw."""
    if ((skopt.__version__, sklearn.__version__, np.__version__) != ("0.10.2", "1.9.1", "1.26.4")
            or not isinstance(seed, (int, np.integer)) or effective_n_jobs(1) != 1):
        return {}
    rng = check_random_state(seed)
    estimator = cook_estimator("ET", space=dimensions, random_state=rng.randint(0, np.iinfo(np.int32).max), n_jobs=1)
    return dict(base_estimator=EnergyForest(**estimator.get_params(deep=False)), random_state=rng)
