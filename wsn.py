import numpy as np
import matplotlib.pyplot as plt
from ialo import ialo

class wsn:
    def __init__(self, l, n, rs, re, params):
        self.l = l
        self.n = n
        self.rs = rs
        self.re = re

        self.alpha1 = params.get('alpha1', 1)
        self.alpha2 = params.get('alpha2', 0)
        self.beta1 = params.get('beta1', 1)
        self.beta2 = params.get('beta2', 2)

        self.grid_resolution = 1.0
        x = np.arange(0, l, self.grid_resolution)
        y = np.arange(0, l, self.grid_resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.grid_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))
        self.total_pixels = len(self.grid_points)

    def calculate_coverage_prob(self, positions):
        sensors = positions.reshape((self.n, 2))

        diff = self.grid_points[:, np.newaxis, :] - sensors[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))

        cond1 = dists <= (self.rs - self.re) # prob = 1
        cond2 = dists >= (self.rs + self.re) # prob = 0
        cond3 = (~cond1) & (~cond2) # prob exponential

        p_cov = np.zeros_like(dists)
        p_cov[cond1] = 1.0
        p_cov[cond2] = 0.0

        if np.any(cond3):
            d_cond3 = dists[cond3]
            lambda1 = self.re - self.rs + d_cond3
            lambda2 = self.re + self.rs - d_cond3
            term = -self.alpha1 * (lambda1 ** self.beta1) / (lambda2 ** self.beta2 + self.alpha2)
            p_cov[cond3] = np.exp(term)

        # eq 3
        joint_prob = 1.0 - np.prod(1.0 - p_cov, axis=1)

        C_th = 0.75
        covered_pixels = np.sum(joint_prob >= C_th)
        R_cov = covered_pixels / self.total_pixels

        return R_cov

    def objective_function(self, positions):
        return -self.calculate_coverage_prob(positions)