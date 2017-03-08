import matplotlib.pyplot as plt


class Plotter:

    def __init__(self):
        pass

    def plot(self, ks, accuracies, x_label, y_label, title):
        plt.plot(ks, accuracies)  # ks are x-axis and accuracies are y-axis
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()
