import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import random


# Step 0: Randomly initialize K means
def initial_guess(iris_copy, k=2, round_places = 2):
    length_max = max(iris_copy[:, 0])
    length_min = min(iris_copy[:, 0])
    width_max = max(iris_copy[:, 1])
    width_min = min(iris_copy[:, 1])

    initial_means = []

    for i in range(k):
        x = random.uniform(length_max, length_min)
        y = random.uniform(width_min, width_max)
        initial_means.append([x, y])

    for i in range(len(initial_means)):
        for j in range(len(initial_means[0])):
            initial_means[i][j] = round(initial_means[i][j], round_places)

    print('initial markers:')
    print(initial_means)

    return initial_means


# Step 1. Assign each data point x_n to it's closest cluster
# NOTE: I use axis distance, not cartesian distance
def cluster_assignment(iris, means):
    n = len(iris)
    k = len(means)
    Z = []

    # for all datapoints (vectors)
    for j in range(n):
        minimum = 1000
        min_index = 0

        # choose argmin ||x_n - mu_i||^2
        for i in range(k):
            single_distance = (iris[j] - means[i]) ** 2
            # Single_distance = vector with axis distances to single mean vector
            if sum(single_distance) < minimum:
                minimum = sum(single_distance)
                min_index = i

        # Single x one-hot vector
        z = np.zeros(len(means))
        z[min_index] = 1

        # All one-hot vectors together
        Z.append(z)
    # change to np.array - to get access to transpose with previous possibility to append :P
    Z = np.array(Z)



    # Return one-hot matrix (if you can call it that)
    # Size of matrix is n x k
    return Z


# Step 2. Recompute the means
def new_mean(iris, Z, round_places=2):
    Z = Z.T
    iris = iris.T
    new_mean_vector = []

    # For each one-hot vector z_i, (i is key in this statement)
    n = len(Z)
    m = len(iris)

    for i in range(n):
        new_mean = []
        bottom = sum(Z[i])

        for j in range(m):
            # Calculate the single coordinate of new mean (average of first iris value)
            new_mean_coordinate = sum(Z[i] * iris[j]) / bottom
            new_mean.append(new_mean_coordinate)

        new_mean_vector.append(new_mean)

    for i in range(len(new_mean_vector)):
        for j in range(len(new_mean_vector[0])):
            new_mean_vector[i][j] = round(new_mean_vector[i][j], round_places)

    print('new mean vector')
    print(new_mean_vector)
    return new_mean_vector


def plot_results(iris, Z, means, index):
    colors = ['b', 'r', 'y', 'g']
    labels = ['first', 'second', 'third', 'fourth']
    n = len(iris)
    k = len(means)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Scatter plot pluses (means positions)
    for i in range(k):
        ax1.scatter(means[i][0], means[i][1], s=100, color=colors[i], marker="+", label=labels[i])

    # Scatter plot datapoints
    for j in range(k):
        pre_data = []
        for i in range(n):
            if Z[j][i] == 1:
                pre_data.append(iris[i])
        pre_data = np.array(pre_data)

        ax1.scatter(pre_data[:, 0], pre_data[:, 1], s=20, color=colors[j], marker='o')

    plt.legend(loc='upper left')
    # print(index)
    name = 'afterplot_' + str(index) + '.jpg'
    plt.savefig(name)


if __name__ == '__main__':

    # Set number of clusters and maximum number of steps
    k = 3
    maxCounter = 10
    round_places = 2

    # Using only petal length and width
    iris = load_iris().data[:, 2:]
    iris_labels = ['petal length', 'petal width']

    if True:
        plt.scatter(iris[:, 0], iris[:, 1], s=20, c='green', marker="o")

        plt.xlabel(iris_labels[0])
        plt.ylabel(iris_labels[1])

        plt.savefig('starting_plot.jpg')

    # Initial cycle
    old_mean = initial_guess(iris, k, round_places=round_places)
    Z = cluster_assignment(iris, old_mean)
    mean = new_mean(iris, Z, round_places=round_places)

    # Conditioned cycle
    Condition = True
    counter = 0
    while Condition:
        # Plotting is always 1 step behind to prevent errors
        plot_results(iris, Z.T, old_mean, counter)

        counter += 1
        old_mean = mean

        Z = cluster_assignment(iris, mean)
        mean = new_mean(iris, Z, round_places=round_places)

        # Check conditions
        if old_mean == mean:
            print('FINISH: Equilibrium reached')
            Condition = False

        elif counter >= maxCounter - 1:
            print('FINISH: Maximum repetitions reached')
            Condition = False

        else:
            for i in range(k):
                if sum(mean[i]) == 0:
                    print('FINISH ERROR: Data point pushed to 0')
                    Condition = False
