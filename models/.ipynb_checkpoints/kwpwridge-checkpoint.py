import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from models.kernels import exp_kernel, abs_kernel

class kwPWRidgeRegressor:
    def __init__(self, alpha=0, bandwidth=None, kernel_func = exp_kernel):
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kernel = kernel_func
        self.models = []
        self.kernel_weights = []

    def fit(self, X, y, split_points):
        X, y = np.array(X), np.array(y)
        
        self.split_points = split_points
        self.split_points.extend([-np.inf, np.inf])
        self.split_points = sorted(list(set(self.split_points)))
        for i in range(len(self.split_points) - 1):
            sr_mask = (X >= self.split_points[max(i - 1, 0)]) & (X <= self.split_points[min(i + 2, len(self.split_points)-1)])
            X_sr, y_sr = X[sr_mask], y[sr_mask]
            r_mask = (X_sr >= self.split_points[i]) & (X_sr <= self.split_points[i + 1])
            X_r, y_r = X_sr[r_mask], y_sr[r_mask]

            if len(X_r) > 0:
                center = np.mean(X_r) #(self.split_points[i] + self.split_points[i + 1]) / 2
                bandwidth = (np.std(X_r) + 0.01) if self.bandwidth is None else self.bandwidth
                kernel_weights = self.kernel(center, bandwidth, X_sr)
                kernel_weights[r_mask] = 1.0
                self.kernel_weights.append(kernel_weights)

                model = Ridge(alpha=self.alpha)
                model.fit(X_sr.reshape(-1, 1), y_sr, sample_weight=kernel_weights)
                self.models.append(model)
            else:
                self.models.append(None)
                
        self.split_points[0] = -np.inf
        self.split_points[-1] = np.inf

    def predict(self, X):
        X = np.array(X)
        y_pred = np.zeros(len(X))
        for i in range(len(self.split_points) - 1):
            mask = (X >= self.split_points[i]) & (X < self.split_points[i + 1])
            if self.models[i] is not None and np.sum(mask) > 0:
                y_pred[mask] = self.models[i].predict(X[mask].reshape(-1, 1))
        return y_pred

class kwPWRidgeRegressorCV:
    def __init__(self, alphas=[0.1, 1, 10, 100], bandwidths=[1, 10, 100], cv=3, random_state=42):
        self.alphas = alphas
        self.bandwidths = bandwidths
        self.cv = cv
        self.random_state = random_state
        self.best_alpha = None
        self.best_bandwidth = None
        self.model = None
        self.scores = []

    def fit(self, X, y, split_points):
        X, y = np.array(X), np.array(y)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        results = []

        for alpha, bandwidth in product(self.alphas, self.bandwidths):
            fold_scores = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model = kwPWRidgeRegressor(alpha=alpha, bandwidth=bandwidth)
                model.fit(X_train, y_train, split_points)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            results.append((avg_score, alpha, bandwidth))
            self.scores.append((alpha, bandwidth, avg_score))

        best_score_index = np.argmax([result[0] for result in results])
        best_score, best_alpha, best_bandwidth = results[best_score_index]
        self.best_alpha = best_alpha
        self.best_bandwidth = best_bandwidth

        self.model = kwPWRidgeRegressor(alpha=self.best_alpha, bandwidth=self.best_bandwidth)
        self.model.fit(X, y, split_points)

    def predict(self, X):
        if self.model is not None:
            return self.model.predict(X)
        else:
            raise Exception("The model has not been fitted yet.")

if __name__ == "__main__":
    
    from sklearn.model_selection import train_test_split
    
    np.random.seed(42)
    X = np.linspace(0, 10, 200)
    y = 4 * np.sin(X) + 0.5 * X**2 + np.random.normal(0, 5, size=200)

    y[50:60] += 15
    y[120:130] -= 10
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    split_points = [3, 6]
    alphas = [1, 10, 100]
    bandwidths = [3, 4]
    cv_model = kwPWRidgeRegressorCV(alphas, bandwidths)
    model = kwPWRidgeRegressor(1)
    cv_model.fit(X_train, y_train, split_points)
    model.fit(X_train, y_train, split_points)
    print(f'Best alpha: {cv_model.best_alpha}, Best bandwidth: {cv_model.best_bandwidth}')
    y_pred_cv = cv_model.predict(X_test)
    y_pred = model.predict(X_test)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_test, y_test, color='blue', label='Data with Discontinuities', alpha=0.6)
    # plt.scatter(X_test, y_pred_cv, color='red', label='CV Piecewise Kernel Weighted Ridge Regression', linewidth=2)
    # plt.scatter(X_test, y_pred, color='green', label='Piecewise Kernel Weighted Ridge Regression', linewidth=2)
    
    plt.plot(X, cv_model.predict(X), color='red', label='CV Piecewise Kernel Weighted Ridge Regression', linewidth=2)
    plt.plot(X, model.predict(X), color='green', label='Piecewise Kernel Weighted Ridge Regression', linewidth=2)
    
    for split in split_points:
        plt.axvline(x=split, color='grey', linestyle='--', linewidth=1)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('CV Piecewise Kernel Weighted Ridge Regression with a Single Split Point')
    plt.legend()
    plt.show()