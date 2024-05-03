import numpy as np

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        self.m = m
        self.reg = reg_param
        self.dim = [m+1, 1]
        self.w = np.zeros(self.dim)

    def gen_poly_features(self, X):
        num_samples, _ = X.shape
        degree = self.m
        poly_features = np.zeros((num_samples, degree+1))
        if degree == 1:
            poly_features[:, 0] = 1
            poly_features[:, 1] = X[:, 0]
        else:
            for i in range(num_samples):
                poly_features[i] = [X[i, 0]**deg for deg in range(degree + 1)]
        return poly_features
    
    def loss_and_grad(self, X, y):
        predictions = self.predict(X)
        poly_features = self.gen_poly_features(X)
        num_samples, _ = X.shape
        loss = np.mean((predictions - y) ** 2)
        gradient = 2 * poly_features.T.dot(predictions - y) / num_samples
        if self.m > 1:
            gradient += self.reg * np.r_[0, self.w[1:]]
        return loss, gradient

    def train_LR(self, X, y, eta=1e-3, batch_size=30, num_iters=1000) :
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
                X_batch = None
                y_batch = None
                batch_indices = np.random.choice(N, batch_size, replace=False)
                for i in range(batch_size):
                    X_batch = X[batch_indices,:]
                    y_batch = y[batch_indices]
               
                loss = 0.0
                grad = np.zeros_like(self.w)
                self.w = self.w.flatten()
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                self.w = self.w - eta*grad
                loss_history.append(loss)
        return loss_history, self.w

    def closed_form(self, X, y):
        poly_features = self.gen_poly_features(X)
        self.w = np.linalg.inv(poly_features.T @ poly_features + self.reg * np.eye(poly_features.shape[1])) @ poly_features.T @ y
        loss, _ = self.loss_and_grad(X, y)
        return loss, self.w
    
    def predict(self, X):
        poly_features = self.gen_poly_features(X)
        predictions = poly_features.dot(self.w).flatten()
        return predictions
