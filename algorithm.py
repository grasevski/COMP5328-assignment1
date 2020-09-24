#!/usr/bin/env python3
"""Run NMF algorithms and calculate evaluation metrics"""
from collections import Counter
import os
from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, normalized_mutual_info_score


def load_data(root: str = 'data/CroppedYaleB',
              reduce: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.
    """
    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):

        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):

            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue

            if not fname.endswith('.pgm'):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')  # grey image.

            # reduce computation complexity.
            img = img.resize([s // reduce for s in img.size])

            # TODO: preprocessing.

            # convert image to numpy array.
            img = np.asarray(img).reshape((-1, 1))

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels


def assign_cluster_label(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Label the data according to clustering, for evaluation"""
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]  # assign label.
    return Y_pred


def plot(V: np.ndarray, V_hat: np.ndarray, V_noise: np.ndarray, red: int,
         imgsize: Tuple[int, int]) -> None:
    """Plot result."""
    img_size = [i // red for i in imgsize]
    ind = 2  # index of demo image
    plt.figure(figsize=(10, 3))
    plt.subplot(131)
    plt.imshow(V_hat[:, ind].reshape(img_size[1], img_size[0]),
               cmap=plt.cm.gray)
    plt.title('Image(Original)')
    plt.subplot(132)
    plt.imshow(V_noise[:, ind].reshape(img_size[1], img_size[0]),
               cmap=plt.cm.gray)
    plt.title('Noise')
    plt.subplot(133)
    plt.imshow(V[:, ind].reshape(img_size[1], img_size[0]), cmap=plt.cm.gray)
    plt.title('Image(Noise)')
    plt.show()


def nmf(Y_hat: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Basic NMF algorithm, using sklearn library"""
    model = NMF(n_components=len(
        set(Y_hat)))  # set n_components to num_classes.
    W = model.fit_transform(V)
    H = model.components_
    return W, H


def zehu4485(Y_hat: np.ndarray,
             V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Johns NMF algorithm"""
    # TODO
    return nmf(Y_hat, V)


def ngra5777(Y_hat: np.ndarray,
             V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nicks NMF algorithm"""
    # TODO
    return nmf(Y_hat, V)


def no_noise(V_hat: np.ndarray) -> np.ndarray:
    """Add no noise"""
    return V_hat


def salt_and_pepper(V_hat: np.ndarray) -> np.ndarray:
    """Randomly change some pixels to black or white"""
    p, r, white = 0.4, 0.3, 255
    p_noise = np.random.rand(*V_hat.shape) <= p
    r_noise = np.random.rand(*V_hat.shape) <= r
    salt = white * p_noise * r_noise
    pepper = -white * p_noise * ~r_noise
    return np.clip(V_hat + salt + pepper, 0, white)


def uniform(V_hat: np.ndarray) -> np.ndarray:
    """Add some uniformly distributed noise"""
    return V_hat + 40 * np.random.rand(*V_hat.shape)


def evaluate_algorithm(
    V: np.ndarray, V_hat: np.ndarray, Y_hat: np.ndarray,
    algorithm: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                        np.ndarray]]
) -> None:
    """Fit model and run evaluation metrics"""
    print(f'==> Apply {algorithm.__name__} NMF ...')
    W, H = algorithm(Y_hat, V)
    print(f'W.shape={W.shape}, H.shape={H.shape}')

    # Evaluate relative reconstruction errors.
    print(f'==> Evaluate RRE ...')
    RRE = np.linalg.norm(V_hat - W.dot(H)) / np.linalg.norm(V_hat)
    print(f'RRE = {RRE}')

    print('==> Evaluate Acc and NMI ...')

    # Assign cluster labels.
    Y_pred = assign_cluster_label(H.T, Y_hat)

    acc = accuracy_score(Y_hat, Y_pred)
    nmi = normalized_mutual_info_score(Y_hat, Y_pred)
    print('Acc(NMI) = {:.4f} ({:.4f})'.format(acc, nmi))


def main():
    """Run all algorithms"""
    for dataset, red, imgsize in ('ORL', 3, (92, 112)), ('CroppedYaleB', 4,
                                                         (168, 192)):
        # Load dataset.
        print(f'==> Load {dataset} dataset ...')
        V_hat, Y_hat = load_data(f'data/{dataset}', red)
        print(f'V_hat.shape={V_hat.shape}, Y_hat.shape={Y_hat.shape}')

        for noise in no_noise, salt_and_pepper, uniform:
            # Add Noise
            print(f'==> Add {noise.__name__} noise ...')
            V = noise(V_hat)

            for algorithm in zehu4485, ngra5777:
                evaluate_algorithm(V, V_hat, Y_hat, algorithm)


if __name__ == '__main__':
    main()
