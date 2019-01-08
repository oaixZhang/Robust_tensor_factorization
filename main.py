import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def loadImages(pa='./att_faces'):
    data = []
    for i in range(1, 41):
        folder = os.path.join(pa, 's%d' % i)
        a = glob.glob(os.path.join(folder, '*.pgm'))
        images = [cv2.imread(d, 0) for d in glob.glob(os.path.join(folder, '*.pgm'))]
        images.append(np.random.randn(112, 92) * 255)
        data.extend(images)
    data = np.array(data)
    print("load image(with noise) shape:", data.shape)
    return data


def twoDPCA(data, l1, l2):
    image_shape = data[0].shape

    mean = np.zeros(image_shape)  # 112*92
    for image in data:
        mean = mean + image
    mean /= len(data)

    ML = np.zeros((image_shape[0], image_shape[0]))
    for image in data:
        diff = image - mean  # centerize
        ML = ML + np.dot(diff, diff.T)
    ML /= len(data)
    L_eval, L_evec = np.linalg.eig(ML)
    sorted_index = np.argsort(L_eval)
    L = L_evec[:, sorted_index[:-l1 - 1: -1]]  # top l1 eigenvectors forms L

    MR = np.zeros((image_shape[1], image_shape[1]))
    for image in data:
        diff = image - mean
        MR += np.dot(diff.T, diff)
    MR = MR / len(data)
    R_eval, R_evec = np.linalg.eig(MR)
    sorted_index = np.argsort(R_eval)
    R = R_evec[:, sorted_index[:-l2 - 1: -1]]  # top l2 eigenvectors forms R

    compressed_images = []
    for image in data:
        cimage = np.dot(np.dot(L.T, image), R)
        compressed_images.append(cimage)
    cimages = np.array(compressed_images)
    return L, R, cimages


def computeErrorFnorm(original, L, R):
    error = 0.0
    for image in original:
        cimage = np.dot(np.dot(L.T, image), R)
        d = np.sum(np.square(image - np.dot(np.dot(L, cimage), R.T)))
        error += d
    return np.sqrt(error / len(original))


def computeErrorR1norm(original, L, R):
    error = 0.0
    for image in original:
        cimage = np.dot(np.dot(L.T, image), R)
        d = np.linalg.norm(image - np.dot(np.dot(L, cimage), R.T), 2)
        error += d
    return error / len(original)


def twoDSVD(data, l1, l2):
    image_shape = data[0].shape
    original_data = data.copy()

    mean = np.zeros(image_shape)  # 112*92
    for image in data:
        mean = mean + image
    mean /= len(data)
    # print("mean: ", mean)
    for image in data:
        image = image - mean
    R = np.eye(92, l2)
    error = []
    for i in range(20):
        ML = np.zeros((image_shape[0], image_shape[0]))
        for image in data:
            ML += np.dot(np.dot(np.dot(image, R), R.T), image.T)
        ML /= len(data)
        L_eval, L_evec = np.linalg.eig(ML)
        sorted_index = np.argsort(L_eval)
        L = L_evec[:, sorted_index[:-l1 - 1: -1]]  # top l1 eigenvectors forms L

        MR = np.zeros((image_shape[1], image_shape[1]))
        for image in data:
            MR += np.dot(np.dot(np.dot(image.T, L), L.T), image)
        MR = MR / len(data)
        R_eval, R_evec = np.linalg.eig(MR)
        sorted_index = np.argsort(R_eval)
        R = R_evec[:, sorted_index[:-l2 - 1: -1]]  # top l2 eigenvectors forms R

        # compute error
        ero = computeErrorFnorm(original_data, L, R)
        error.append(ero)
        if i == 0:
            continue
        if (np.abs(error[i - 1] - error[i])) < 0.05:
            break
    print("2DSVD, i=%d ,error:" % len(error), error)
    compressed_images = []
    for image in data:
        cimage = np.dot(np.dot(L.T, image), R)
        compressed_images.append(cimage)
    cimages = np.array(compressed_images)
    return L, R, cimages, error


def robustTwoDSVD(data, l1, l2):
    image_shape = data[0].shape
    original_data = data.copy()

    mean = np.zeros(image_shape)  # 112*92
    for image in data:
        mean = mean + image
    mean /= len(data)
    for image in data:
        image = image - mean  # centerize

    # init R and L
    R = np.eye(92, l2)
    ML = np.zeros((image_shape[0], image_shape[0]))
    for image in data:
        ML += np.dot(np.dot(np.dot(image, R), R.T), image.T)
    ML /= len(data)
    L_eval, L_evec = np.linalg.eig(ML)
    sorted_index = np.argsort(L_eval)
    L = L_evec[:, sorted_index[:-l1 - 1: -1]]  # top l1 eigenvectors forms L
    rs = []
    for image in data:
        ri = np.sqrt(np.abs(np.trace(
            np.dot(image.T, image) - 2 * np.dot(np.dot(np.dot(np.dot(np.dot(image.T, L), L.T), image), R), R.T))))
        rs.append(ri)
    cutoff = np.median(np.array(rs))
    # end of init

    error = []
    for i in range(20):
        MR = np.zeros((image_shape[1], image_shape[1]))
        for image in data:
            ri = np.sqrt(np.abs(np.trace(
                np.dot(image.T, image) - 2 * np.dot(np.dot(np.dot(np.dot(np.dot(image.T, L), L.T), image), R), R.T))))
            if ri < cutoff:
                MR += np.dot(np.dot(np.dot(image.T, L), L.T), image)
            else:
                MR += (cutoff / ri) * np.dot(np.dot(np.dot(image.T, L), L.T), image)
        MR = MR / len(data)
        R_eval, R_evec = np.linalg.eig(MR)
        sorted_index = np.argsort(R_eval)
        R = R_evec[:, sorted_index[:-l2 - 1: -1]]  # top l2 eigenvectors forms R
        ML = np.zeros((image_shape[0], image_shape[0]))
        for image in data:
            ri = np.sqrt(np.abs(np.trace(
                np.dot(image.T, image) - 2 * np.dot(np.dot(np.dot(np.dot(np.dot(image.T, L), L.T), image), R), R.T))))
            if ri < cutoff:
                ML += np.dot(np.dot(np.dot(image, R), R.T), image.T)
            else:
                ML += (cutoff / ri) * np.dot(np.dot(np.dot(image, R), R.T), image.T)
        ML /= len(data)
        L_eval, L_evec = np.linalg.eig(ML)
        sorted_index = np.argsort(L_eval)
        L = L_evec[:, sorted_index[:-l1 - 1: -1]]  # top l1 eigenvectors forms L
        # compute error
        ero = computeErrorR1norm(original_data, L, R)
        error.append(ero)
        if i == 0:
            continue
        if (np.abs(error[i - 1] - error[i])) < 0.05:
            break
    print("robust, i=%d ,error:" % len(error), error)
    compressed_images = []
    for image in data:
        cimage = np.dot(np.dot(L.T, image), R)
        compressed_images.append(cimage)
    compressed_images = np.array(compressed_images)
    return L, R, compressed_images, error


if __name__ == "__main__":
    np.random.seed(1234)
    data = loadImages()
    L1 = 20
    L2 = 20
    # 2DPCA
    L, R, cimages_2DPCA = twoDPCA(data, L1, L2)
    rimage_2DPCA = []
    for cimage in cimages_2DPCA:
        rimage = np.dot(np.dot(L, cimage), R.T)
        rimage_2DPCA.append(rimage)
    rimage_2DPCA = np.array(rimage_2DPCA)

    # Robust_tensor_factorization
    t1 = time.time()
    L_2DSVD, R_2DSVD, cimages_2DSVD, ero_2DSVD = twoDSVD(data, L1, L2)
    rimages_2DSVD = []
    for cimage_2DSVD in cimages_2DSVD:
        rimage_2DSVD = np.dot(np.dot(L_2DSVD, cimage_2DSVD), R_2DSVD.T)
        rimages_2DSVD.append(rimage_2DSVD)
    rimages_2DSVD = np.array(rimages_2DSVD)
    t_2DSVD = time.time() - t1
    print("time of 2DSVD: ", t_2DSVD)

    # robust Robust_tensor_factorization
    t2 = time.time()
    L_robust, R_robust, cimages_robust, ero_robust = robustTwoDSVD(data, L1, L2)
    rimages_robust = []
    for cimage_robust in cimages_robust:
        rimage_robust = np.dot(np.dot(L_robust, cimage_robust), R_robust.T)
        rimages_robust.append(rimage_robust)
    rimages_robust = np.array(rimages_robust)
    t_robust = time.time() - t1
    print("time of robust: ", t_robust)

    for i in range(1, 7):
        plt.subplot(4, 6, i)
        plt.title("original")
        plt.imshow(data[i * 10], cmap='gray')
        plt.axis('off')
        plt.subplot(4, 6, i + 6 * 1)
        plt.title("2DPCA")
        plt.imshow(rimage_2DPCA[i * 10], cmap='gray')
        plt.axis('off')
        plt.subplot(4, 6, i + 6 * 2)
        plt.title("2DSVD")
        plt.imshow(rimages_2DSVD[i * 10], cmap='gray')
        plt.axis('off')
        plt.subplot(4, 6, i + 6 * 3)
        plt.title("robust")
        plt.imshow(rimages_robust[i * 10], cmap='gray')
        plt.axis('off')
    plt.show()
    plt.subplot(121)
    plt.title("2DSVD")
    plt.plot(ero_2DSVD)
    plt.subplot(122)
    plt.title("robust")
    plt.plot(ero_robust)
    plt.show()
