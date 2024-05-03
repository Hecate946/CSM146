import numpy as np


class Logistic(object):
    def __init__(self, d=784, reg_param=0):
        """ "
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg = reg_param
        self.dim = [d + 1, 1]
        self.w = np.zeros(self.dim)

    def gen_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N, d = X.shape
        X_out = np.zeros((N, d + 1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        for i in range(N):
            X_out[i] = np.append(1, X[i])
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out

    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels
        Returns:
        - loss: a real number represents the loss
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w
        """
        loss = 0.0
        grad = np.zeros_like(self.w)
        N, d = X.shape

        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #
        # Calculate the probability that each class label is 1
        # Add a column of ones to X to account for the bias term
        # X_bias = np.hstack([np.ones((N, 1)), X])
        
        # Calculate the probability that each class label is 1
        X_out = self.gen_features(X)
        running_sum = 0
        W = self.w.flatten()
        grad = grad.flatten()
        for i in range(N):
            y_i = (y[i]+1)/2
            x_i = X_out[i]
            h = np.dot(W, x_i)
            running_sum += np.log(1+np.exp(h))
            if y_i == 1:
                running_sum -= h
            
            sigmoid = 1/(1+np.exp(-h))
            grad = grad + (sigmoid-y_i)*x_i/N
        loss = running_sum/N
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad

    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000):
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights
        """
        loss_history = []
        N, d = X.shape
        for t in np.arange(num_iters):
            X_batch = None
            y_batch = None
            # ================================================================ #
            # YOUR CODE HERE:
            # Sample batch_size elements from the training data for use in gradient descent.
            # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
            # The indices should be randomly generated to reduce correlations in the dataset.
            # Use np.random.choice.  It is better to user WITHOUT replacement.
            # ================================================================ #
            batch_indices = np.random.choice(N, batch_size, replace=False)
            for i in range(batch_size):
                X_batch = X[batch_indices,:]
                y_batch = y[batch_indices]           

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss = 0.0
            grad = np.zeros_like(self.w)
            # ================================================================ #
            # YOUR CODE HERE:
            # evaluate loss and gradient for batch data
            # save loss as loss and gradient as grad
            # update the weights self.w
            # ================================================================ #
            loss, grad = self.loss_and_grad(X_batch, y_batch)
            self.w = self.w - eta*grad
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss_history.append(loss)
        return loss_history, self.w

    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labelss for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        X_out = self.gen_features(X)
        self.w = self.w.flatten()
        for i in range(X_out.shape[0]):
            wx = np.dot(self.w, X_out[i])
            h = 1/(1+np.exp(-wx))
            if h > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = -1

        y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred