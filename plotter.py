import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self):
        self.x_data = []
        self.y_data = []

    def add_x_data(self, x):
        self.x_data.append(float(x))

    def add_y_data(self, y):
        self.y_data.append(float(y))

    def plot(self, x_label="", y_label="", title=""):
        plt.plot(self.x_data, self.y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.suptitle(title)
        plt.title("Mean = " + str(np.mean(self.y_data)) + ", Std. = " + str(np.std(self.y_data)))
        plt.show()
