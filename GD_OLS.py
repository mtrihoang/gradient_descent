import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GD_basic:
    """The class is created to simulate the Gradient Descent algorithm."""

    def __init__(self, theta, beta, X, y, eta):
        """Update attributes of the class.

        Args:
        theta (numpy.ndarray): The initial guess for OLS coefficients.
        beta (numpy.ndarray): The true OLS parameters.
        X (numpy.ndarray): The matrix of covariates.
        y (numpy.ndarray): The response variable.
        eta (float): The learning rate.

        """
        self.theta = theta
        self.beta = beta
        self.X = X
        self.y = y
        self.eta = eta

    def grad_vec(self):
        """Update gradients for OLS."""
        return 2 / self.X.shape[0] * self.X.T @ (self.X @ self.theta - self.y)

    def fixed_point(self):
        """Find the global optimum of an objective function."""
        theta_list = [self.theta]
        grad = GD_basic.grad_vec(self)
        while np.linalg.norm(grad, 2) > 1e-3:
            self.theta = self.theta - self.eta * grad
            grad = GD_basic.grad_vec(self)
            theta_list.append(self.theta)
        return theta_list


def mse_example(X, y, theta1_new, theta2_new, N):
    """Create the functional form of MSE."""
    return (
        1
        / N
        * (
            np.linalg.norm(y, 2) ** 2
            + np.linalg.norm(X[:, 0], 2) ** 2 * theta1_new**2
            + np.linalg.norm(X[:, 1], 2) ** 2 * theta2_new**2
            + 2 * X[:, 0] @ X[:, 1] * theta1_new * theta2_new
            - 2 * X[:, 0] @ y * theta1_new
            - 2 * X[:, 1] @ y * theta2_new
        )
    )


def plot_example(results, mse, c_num, time):
    """The function is to create a dynamic graph to show
    the convergence to a local optimum.
        Args:
        results (list): The list of solution candidates for OLS coefficients.
        mse (numpy.ndarray): The values of MSE.
        c_num (integer): The number of contours.
        time (float): The animation duration of the .gif file.

    Returns:
    -------
        The dynamic graph associated with the convergence of theta.
    """
    c_levels = np.linspace(np.min(mse), np.max(mse), c_num)

    list_frames = []

    fig, ax = plt.subplots()
    contours = plt.contour(theta1_new, theta2_new, mse, colors="black", levels=c_levels)
    ax.clabel(contours, inline=True, fontsize=10)
    ax.plot(results[-1][0], results[-1][1], "o:r", markersize=7)

    for i in range(len(results) - 1):
        ax.plot(results[i][0], results[i][1], "o:b", markersize=5)
        ax.plot(results[i + 1][0], results[i + 1][1], "o:b", markersize=5)
        ax.plot(
            [results[i][0], results[i + 1][0]], [results[i][1], results[i + 1][1]], "b"
        )
        ax.set_title(
            f"Iteration: {i+1}, theta1 = {results[i][0].round(3)}, \
                theta2 = {results[i][1].round(3)}"
        )
        ax.set_xlabel("theta1")
        ax.set_ylabel("theta2")
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        list_frames.append(Image.fromarray(frame))

    plt.close(fig)

    list_frames[0].save(
        "GD_without_momentum.gif",
        save_all=True,
        append_images=list_frames[1:],
        duration=time * 1000,
        loop=0,
    )


if __name__ == "__main__":
    np.random.seed(123)

    N = 1000
    mean = [1, 1]
    cov_matrix = np.array([[1, 0], [0, 1]])
    X = np.random.multivariate_normal(mean, cov_matrix, N)
    epsilon = np.random.normal(0, 1, (N, 1))
    beta = np.array([-1, 1]).reshape(-1, 1)
    y = X @ beta + epsilon
    theta = np.array([6, -4]).reshape(-1, 1)

    SGD_example = GD_basic(theta, beta, X, y, 0.3)
    results = SGD_example.fixed_point()

    theta1 = np.arange(-7, 7, 0.01)
    theta2 = np.arange(-7, 7, 0.01)
    theta1_new, theta2_new = np.meshgrid(theta1, theta2)
    mse = mse_example(X, y, theta1_new, theta2_new, N)

    plot_example(results, mse, 30, 0.15)
