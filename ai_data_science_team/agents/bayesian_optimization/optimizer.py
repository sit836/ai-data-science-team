from typing import List

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class BayesianOptimizer:
    """贝叶斯优化器实现"""

    def __init__(self, bounds: List[tuple], n_initial_points: int = 5):
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.X = []
        self.y = []
        self.gp = None

    def initialize(self, X_init, y_init):
        """初始化优化器"""
        self.X = X_init
        self.y = y_init

    def fit_gp(self):
        """拟合高斯过程"""
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        X_array = np.array(self.X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        self.gp.fit(X_array, np.array(self.y))

    def acquisition_function(self, x, xi: float = 0.01):
        """获取函数（预期改进）"""
        if len(self.y) == 0:
            return 0

        x = np.array(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        mu, sigma = self.gp.predict(x, return_std=True)
        if sigma == 0:
            return 0

        best_y = max(self.y)
        z = (mu - best_y - xi) / sigma
        return (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    def suggest_next_point(self):
        """建议下一个采样点"""
        if len(self.X) < self.n_initial_points:
            return [np.random.uniform(b[0], b[1]) for b in self.bounds]

        self.fit_gp()

        def neg_acquisition(x):
            return -self.acquisition_function(x)

        result = minimize(
            neg_acquisition,
            x0=[np.random.uniform(b[0], b[1]) for b in self.bounds],
            bounds=self.bounds,
            method="L-BFGS-B",
        )

        return result.x.tolist()

    def update(self, x, y):
        """更新优化器"""
        self.X.append(x)
        self.y.append(y)

