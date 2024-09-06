__author__ = 'Jackson Eshbaugh'
__date__ = 'September 6, 2024'

'''
An implementation of a k-means cluster that reduces the number of colors in an image.
Parts of this code adapted from Sebastian Charmot's article:
https://towardsdatascience.com/clear-and-visual-explanation-of-the-k-means-algorithm-applied-to-image-compression-b7fdc547e410
'''
import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


def init_centroids(num_clusters, image):
    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    centroids_init = np.empty([num_clusters, 3])

    for i in range(num_clusters):
        rand_row = random.randint(0, H)
        rand_col = random.randint(0, W)
        centroids_init[i] = image[rand_row, rand_col]

    return centroids_init


def update_centroids(centroids, image, max_iter=30):
    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    for i in range(max_iter):
        centroid_rgbs = {}

        for row in range(H):
            for col in range(W):
                centroid = np.argmin(np.linalg.norm(centroids - image[row, col], axis=1))
                if centroid in centroid_rgbs:
                    centroid_rgbs[centroid] = np.append(centroid_rgbs[centroid], [image[row, col]], axis=0)
                else:
                    centroid_rgbs[centroid] = np.array([image[row, col]])

        for k in centroid_rgbs:
            centroids[k] = np.mean(centroid_rgbs[k], axis=0)

    return centroids


def update_image(image, centroids):
    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    for row in range(H):
        for col in range(W):
            nearest_centroid = np.argmin(np.linalg.norm(centroids - image[row, col], axis=1))
            image[row, col] = centroids[nearest_centroid]

    return image


if __name__ == "__main__":

    k = 16

    input_file = input('Input File (include extension): ')
    output_file = input('Output File (no extension, will be png): ')
    new_k = input('k (or just press enter to continue with default): ')

    if (new_k != '') or (is_number(new_k)):
        k = new_k

    image = np.copy(mpimg.imread(input_file))

    initial_centroids = init_centroids(k, image)
    final_centroids = update_centroids(initial_centroids, image, max_iter=30)
    image_compressed = update_image(image, final_centroids)

    plt.imshow(image_compressed)
    plt.savefig(fname='f{output_file}.png', format='png', dpi=300)
