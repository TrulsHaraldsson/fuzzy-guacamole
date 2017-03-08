import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self):
        pass

    def plot(self, ks, accuracies, x_label, y_label, title):
        plt.plot(ks, accuracies)  # ks are x-axis and accuracies are y-axis
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.suptitle(title)
        plt.title("Mean Acc. = " + str(np.mean(accuracies)) + ", Std. Acc. " + str(np.std(accuracies)))
        plt.show()