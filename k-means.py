#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from matplotlib import pyplot as plt


class Coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def load_data(file_name):
    data_list = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            cur_line = line.strip().split()
            data_list.append(Coordinate(float(cur_line[0]), float(cur_line[1])))

    return data_list


def dist(a, b):
    result = math.sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))
    return result


def compute_centroids(data_list, centroids_num, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(centroids_num):
        groups.append([])
        new_centroids.append(Coordinate(0, 0))

    for data in data_list:
        distance_list = []
        for centroid in centroids:
            distance = dist(data, centroid)
            distance_list.append(distance)

        distance_index = distance_list.index(min(distance_list))
        groups[distance_index].append(data)
        loss += min(distance_list)
        new_centroids[distance_index].x += data.x
        new_centroids[distance_index].y += data.y

    for i in range(centroids_num):
        new_centroids[i].x /= len(groups[i])
        new_centroids[i].y /= len(groups[i])

    return new_centroids, groups, loss


def draw(centroids, groups, centroids_num):
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, centroids_num)]
    for k, centroid in enumerate(centroids):
        x_list = []
        y_list = []
        for g in groups[k]:
            x_list.append(g.x)
            y_list.append(g.y)
        plt.plot(y_list, x_list, '.', markerfacecolor=tuple(colors[k]), markersize=1.2)

    plt.show()


def k_means(data_list, centroids_num, convergence, iters_num):

    centroid_indices = np.random.choice(len(data_list), centroids_num)
    centroids = []
    for centroid_index in centroid_indices:
        centroids.append(data_list[centroid_index])

    centroids, groups, old_loss = compute_centroids(data_list, centroids_num, centroids)
    iterations = 1
    while True:
        centroids, groups, loss = compute_centroids(data_list, centroids_num, centroids)
        iterations = iterations + 1
        print("loss = %f" % loss)
        if abs(old_loss - loss) < convergence or iterations > iters_num:
            break
        old_loss = loss
        for centroid in centroids:
            print(centroid.x, centroid.y)

    # print result
    print("k-means result：\n")
    for centroid in centroids:
        print(centroid.x, centroid.y)

    draw(centroids, groups, centroids_num)


def main():
    data = load_data('julei1.txt')
    centroids_num = 2
    loss_convergence = 1e-6
    iterations_num = 100
    k_means(data, centroids_num, loss_convergence, iterations_num)


if __name__ == '__main__':
    main()
