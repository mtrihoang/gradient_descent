import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from PIL import Image


class GD_sim:
    """The class is created to simulate the Gradient Descent with Momentum algorithm."""

    def __init__(self, func, theta, gamma, eta, v):
        """Update attributes of the class."""
        self.func = func
        self.theta = theta
        self.gamma = gamma
        self.eta = eta
        self.v = v

    def GD_with_momentum(self):
        """The method is created to find global optimum of an objective function.

        Returns:
        -------
            theta_vec (list): The sequence of numbers associated with the direction of
            steepest descent in which the local optimum is the last element of the list.

        """
        theta_vec = []
        nabla = derivative(self.func, self.theta)
        while abs(nabla) > 1e-3:
            self.v = self.gamma * self.v + self.eta * nabla
            self.theta = self.theta - self.v
            nabla = derivative(self.func, self.theta)
            theta_vec.append(self.theta)
        return theta_vec


def plot_example(x, func, theta, gamma, eta, v, time):
    """ "The function is to create a dynamic graph to show
    the convergence to a local optimum.
        Args:
        func (function): The objective function.
        theta: The initial point or the first guess of local optimum.
        gamma: The decay factor.
        eta: The learning rate.
        v: The momentum.
        time: The animation duration of the .gif file.

    Returns:
    -------
        The dynamic graph associated with the convergence of theta.
    """
    list_frames = []

    GD_example = GD_sim(func, theta, gamma, eta, v)
    results = GD_example.GD_with_momentum()
    for i in range(len(results) - 1):
        fig, ax = plt.subplots()
        y = func(x)
        ax.plot(x, y, "b")
        x_old, x_new = results[i], results[i + 1]
        y_old, y_new = func(x_old), func(x_new)
        ax.plot(x_old, y_old, "ko")
        ax.plot(x_new, y_new, "ro")
        ax.plot([x_old, x_new], [y_old, y_new], "k")
        ax.set_title(f"Iteration: {i+1}, x = {round(x_new, 1)}, y = {round(y_new, 1)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        list_frames.append(Image.fromarray(frame))
        plt.close(fig)

    list_frames[0].save(
        "GD_with_momentum.gif",
        save_all=True,
        append_images=list_frames[1:],
        duration=time * 1000,
        loop=0,
    )


if __name__ == "__main__":

    def function_example(x):
        return np.sin(x) + np.cos(x / 2)

    plot_example(np.linspace(-12, 12, 1000), function_example, 1, 0.5, 0.9, 0, 0.15)
    plot_example(np.linspace(-12, 12, 1000), function_example, 1, 0.5, 0.1, 0, 0.15)

    GD_example = GD_sim(function_example, 5, 0.9, 0.1, 0)
    results = GD_example.GD_with_momentum()
