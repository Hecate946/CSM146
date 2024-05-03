import matplotlib.pyplot as plt
import numpy as np


def perceptron(X, y, epochs=1):
    """
    Implementation of the perceptron algorithm with:
    - Learning rate (eta) = 1
    - No bias term
    - Specified number of epochs
    """
    w = X[0]  # Initialize weights to the first sample
    for epoch in range(epochs):
        for i in range(len(X)):
            a = np.dot(w, X[i])
            prediction = 1 if a == 0 else np.sign(a)
            if prediction != y[i]:
                w = w + y[i] * X[i]
                
            print(f"Epoch {epoch+1}, Update {i+1}: w = {w}")
            yield w

    return w


def plot_points(X, y):
    # Split the data based on class
    X_pos = X[y == 1]
    X_neg = X[y == -1]

    # Plot the points
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c="teal", s=100, label="y = 1")
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c="purple", s=100, label="y = -1")
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Data Point Classification")
    plt.legend(["y = 1", "y = -1"], loc="upper left")

    plt.savefig('dataset.png')

def plot_perceptron(X, y, epochs=1):
    # Split the data based on class
    X_pos = X[y == 1]
    X_neg = X[y == -1]
    # Plot points
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c="teal", s=100, label="y = 1")
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c="purple", s=100, label="y = -1")

    plt.ylim(-3, 3)
    plt.grid(True)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(["y = 1", "y = -1"], loc="upper left")

    for w in perceptron(X, y, epochs):
        if w[1] == 0:  # Prevent division by zero if w[1] becomes zero
            continue

        slope = -w[0] / w[1]
        x_vals = np.linspace(-2, 5, 100)
        y_vals = slope * x_vals

        plt.plot(x_vals, y_vals)

    plt.title("Visualization of Perceptron Learning Algorithm")
    plt.savefig("perceptron.png")

def count_errors():
    weight_vectors = [np.array([4, 0]), np.array([3, -1]), np.array([1, -3]), np.array([-2, -3])]
    votes = [2, 2, 4, 1]

    X = np.array([[4, 0], [1, 1], [0, 1], [-2, -2], [-2, 1], [1, 0], [5, 2], [3, 0]])
    y = np.array([1, -1, -1, 1, -1, 1, -1, -1])

    perceptron_predictions = np.sign(np.dot(X, [-2, -3]))
    perceptron_errors = np.sum(perceptron_predictions != y)

    voted_perceptron_predictions = np.sign(np.sum([v * np.sign(np.dot(X, w)) for w, v in zip(weight_vectors, votes)], axis=0))
    voted_perceptron_errors = np.sum(voted_perceptron_predictions != y)

    average_perceptron_predictions = np.sign(np.sum([v * np.dot(X, w) for w, v in zip(weight_vectors, votes)], axis=0))
    average_perceptron_errors = np.sum(average_perceptron_predictions != y)

    perceptron_error_rate = perceptron_errors / len(X)
    voted_perceptron_error_rate = voted_perceptron_errors / len(X)
    average_perceptron_error_rate = average_perceptron_errors / len(X)

    print(f"Perceptron Predictions: {perceptron_predictions}")
    print(f"Voted Perceptron Predictions: {voted_perceptron_predictions}")
    print(f"Average Perceptron Predictions: {average_perceptron_predictions}\n")
    
    print(f"Perceptron Error Rate: {perceptron_error_rate}")
    print(f"Voted Perceptron Error Rate: {voted_perceptron_error_rate}")
    print(f"Average Perceptron Error Rate: {average_perceptron_error_rate}")






if __name__ == "__main__":
    # Data: feature vectors and labels
    X = np.array([[4, 0], [1, 1], [0, 1], [-2, -2], [-2, 1], [1, 0], [5, 2], [3, 0]])
    y = np.array([1, -1, -1, 1, -1, 1, -1, -1])

    plot_points(X, y)

    epochs = 1
    perceptron(X, y, epochs)
    plot_perceptron(X, y, epochs)

    count_errors()
