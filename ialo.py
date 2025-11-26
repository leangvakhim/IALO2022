import numpy as np
import math
import random
from tqdm import tqdm

class ialo:
    def __init__(self, obj_func, dim, lb, ub, pop_size, max_iter):
        self.objc_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

        # handle bounds being scalar or vector
        self.lb = np.array(lb) if isinstance(lb, list) else np.full(dim, lb)
        self.ub = np.array(ub) if isinstance(ub, list) else np.full(dim, ub)

        # IALO parameters
        self.cr = 0.95
        self.f = 0.5
        self.beta = 1.5

        self.antlions = np.zeros((pop_size, dim))
        self.antlions_fitness = np.zeros(pop_size)
        self.ants = np.zeros((pop_size, dim))
        self.ants_fitness = np.zeros(pop_size)

        self.elite = None
        self.elite_fitness = float('inf')

    def init_population(self):
        self.antlions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.ants = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

        for i in range(self.pop_size):
            self.antlions_fitness[i] = self.objc_func(self.antlions[i])
            if self.antlions_fitness[i] < self.elite_fitness:
                self.elite_fitness = self.antlions_fitness[i]
                self.elite = self.antlions[i].copy()

    def levy_flight(self, beta):
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, sigma_v, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def cauchy_mutation(self):
        # eq 19-20
        return np.random.standard_cauchy(self.dim)

    def roulette_wheel_selection(self):
        fitness = self.antlions_fitness
        if np.min(fitness) < 0:
            fitness = fitness - np.min(fitness) + 1e-5

        accumulation = np.cumsum(1.0 / (fitness + 1e-10))
        p = np.random.rand() * accumulation[-1]
        idx = np.searchsorted(accumulation, p)
        return idx

    def optimize(self):
        self.init_population()

        convergence_curve = []

        for t in tqdm(range(self.max_iter), desc="IALO Progress: "):
            # eq 10
            if t > 0.95 * self.max_iter: w = 6
            elif t > 0.90 * self.max_iter: w = 5
            elif t > 0.75 * self.max_iter: w = 4
            elif t > 0.50 * self.max_iter: w = 3
            elif t > 0.10 * self.max_iter: w = 2
            else: w = 1

            I = 1 + (10 * w) ** (t / self.max_iter)

            for i in range(self.pop_size):
                rolette_index = self.roulette_wheel_selection()
                selected_antlion = self.antlions[rolette_index]

                c = self.lb / I
                d = self.ub / I

                ra_step = (np.random.rand(self.dim) * (d - c) + c) + selected_antlion
                re_step = (np.random.rand(self.dim) * (d - c) + c) + self.elite

                new_ant_pos = (ra_step + re_step) / 2

                p = np.random.rand()

                old_ant_pos = self.ants[i].copy()
                old_ant_fit = self.ants_fitness[i] if t > 0 else float('inf')

                if p < 0.5:
                    # eq 18
                    alpha0 = 0.01
                    step_size = alpha0 * (new_ant_pos - self.elite) * self.levy_flight(self.beta)
                    new_ant_pos = new_ant_pos + step_size
                else:
                    new_ant_pos = new_ant_pos + new_ant_pos * self.cauchy_mutation()

                new_ant_pos = np.clip(new_ant_pos, self.lb, self.ub)

                new_fitness = self.objc_func(new_ant_pos)

                # eq 22
                if new_fitness < old_ant_fit or t == 0:
                    self.ants[i] = new_ant_pos
                    self.ants_fitness[i] = new_fitness
                else:
                    pass

            # eq 13
            combined_pop = np.vstack((self.antlions, self.ants))
            combined_fit = np.concatenate((self.antlions_fitness, self.antlions_fitness))

            sorted_indices = np.argsort(combined_fit)

            self.antlions = combined_pop[sorted_indices[: self.pop_size]]
            self.antlions_fitness = combined_fit[sorted_indices[:self.pop_size]]

            if self.antlions_fitness[0] < self.elite_fitness:
                self.elite_fitness = self.antlions_fitness[0]
                self.elite = self.antlions[0].copy()

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                # eq 23
                v = self.antlions[r1] + self.f * (self.antlions[r2] - self.antlions[r3])

                # eq 24
                u = np.zeros(self.dim)
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() <= self.cr or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = self.antlions[i][j]

                u = np.clip(u, self.lb, self.ub)

                # eq 25
                fit_u = self.objc_func(u)
                if fit_u < self.antlions_fitness[i]:
                    self.antlions[i] = u
                    self.antlions_fitness[i] = fit_u

            current_best_idx = np.argmin(self.antlions_fitness)
            if self.antlions_fitness[current_best_idx] < self.elite_fitness:
                self.elite_fitness = self.antlions_fitness[current_best_idx]
                self.elite = self.antlions[current_best_idx]

            convergence_curve.append(self.elite_fitness)

        return self.elite, self.elite_fitness, convergence_curve