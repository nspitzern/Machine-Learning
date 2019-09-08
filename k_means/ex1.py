import init_centroids
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math

from scipy.misc import imread


def read_image():
    #read the image
    path = 'dog.jpeg'
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    return X, img_size


def show_image(A_norm, img_size):
    X = A_norm.reshape((img_size[0], img_size[1], img_size[2]))
    # plot the image
    plt.imshow(X)
    plt.grid(False)
    plt.show()


def loss_plot(loss_array):
    plt.plot(loss_array)
    plt.grid(False)
    plt.show()


def calculate_distance(pixel1, pixel2):
    # sqrt(x^2+y^2+z^2)
    return np.linalg.norm(pixel1 - pixel2, axis=1)**2


def classify_pixel(pixel, centroids):
    cent_index = np.argmin(calculate_distance(pixel, centroids), axis=0)
    return cent_index


def calculate_new_centroids(centroids, num_of_accumulated_pixels):
    for l in range(len(num_of_accumulated_pixels)):
        if num_of_accumulated_pixels[l] != 0:
            centroids[l] = centroids[l] / num_of_accumulated_pixels[l]
    return centroids


def print_centroids(j, centroids):
    print("iter %d: " % j, end='')
    print(print_cent(centroids))


def accumulate_centroids(pixel, centroids, new_centroids):
    cent_index = classify_pixel(pixel, centroids)
    new_centroids[cent_index] += pixel
    return new_centroids, cent_index


def assign_centroids_to_image(centroids, img, img_size):
    for i, pixel in enumerate(img):
        index = classify_pixel(pixel, centroids)
        img[i] = centroids[index]
    show_image(img, img_size)


def calculate_loss(centroids, img):
    loss = 0
    for pixel in img:
        cent_index = classify_pixel(pixel, centroids)
        loss += np.linalg.norm(pixel - centroids[cent_index], axis=0)**2
    return loss


def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]


def main():
    img, img_size = read_image()
    ## show_image(img, img_size)

    i = [2, 4, 8, 16]
    for k in i:
        print("k=%d:" % k)
        centroids = init_centroids.init_centroids(img, k)
        loss_array = []
        # do 10 iterations
        for j in range(11):
            # calculate the loss
            ## loss = calculate_loss(centroids, img.copy())
            ## loss_array.append(loss)
            # create an array for the new centroids
            new_centroids = np.zeros((k, 3))
            # print the centroids
            print_centroids(j, centroids.copy())
            # create index vector to count points for each centroid
            num_of_accumulated_pixels = np.zeros(k)
            # go over the image
            for pixel in img:
                # classify the pixel
                new_centroids, index = accumulate_centroids(pixel, centroids, new_centroids)
                # add 1 to the centroid index
                num_of_accumulated_pixels[index] += 1
            # calculate the new centroids
            centroids = calculate_new_centroids(new_centroids, num_of_accumulated_pixels)
        # plot the loss
        ## loss_plot(np.asarray(loss_array))
        ## loss_array.clear()
        ## assign_centroids_to_image(centroids, img.copy(), img_size)


main()

