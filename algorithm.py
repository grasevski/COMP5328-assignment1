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
from sklearn.model_selection import train_test_split


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

            # convert image to numpy array.
            img = np.asarray(img).reshape((-1, 1)) / 255

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels


def assign_cluster_label(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Label the data according to clustering, for evaluation"""
    kmeans = KMeans(n_clusters=len(set(Y)), random_state=0).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]  # assign label.
    return Y_pred


def plot(red: int, imgsize: Tuple[int, int], *images: np.ndarray) -> None:
    """Plot a list of images side by side"""
    img_size = [i // red for i in imgsize]
    ind = 2  # index of demo image
    plt.figure(figsize=(10, 3))
    for i, x in enumerate(images, 1):
        plt.subplot(1, len(images), i)
        plt.imshow(x[:, ind].reshape(img_size[1], img_size[0]),
                   cmap=plt.cm.gray)
    plt.show()


def nmf(K: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Basic NMF algorithm, using sklearn library"""
    model = NMF(n_components=K, random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    return W, H


def nmf_beta(K: int,
             X: np.ndarray,
             beta: float = 2,
             l1: float = 0,
             l2: float = 0,
             steps: int = 200,
             tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Algorithms for nonnegative matrix factorization with the B divergence"""
    rng, avg = np.random.RandomState(0), np.sqrt(X.mean() / K)
    W, H = avg * rng.rand(len(X), K), avg * rng.rand(K, len(X[0]))
    error_at_init = rre(X, W, H)
    previous_error = error_at_init
    for _ in range(steps):
        W *= (X * ((W @ H)**(beta - 2))) @ H.T / ((
            (W @ H)**(beta - 1)) @ H.T + l1 + l2 * W)
        H *= W.T @ (X * ((W @ H)**(beta - 2))) / (W.T @ (
            (W @ H)**(beta - 1)) + l1 + l2 * H)
        error = rre(X, W, H)
        if (previous_error - error) / error_at_init < tol:
            break
        previous_error = error
    return W, H


def tanh_nmf(K: int,
             X: np.ndarray,
             p: float = 1,
             b: float = 1e-2,
             y: float = 0,
             steps: int = 200,
             tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Another Robust NMF"""
    rng, avg = np.random.RandomState(0), np.sqrt(X.mean() / K)
    W, H = avg * rng.rand(len(X), K), avg * rng.rand(K, len(X[0]))
    D = np.zeros(H.shape)
    error_at_init = rre(X, W, H)
    previous_error = error_at_init
    for _ in range(steps):
        if y:
            for i in range(len(D)):
                for j in range(len(D[0])):
                    D[i][j] = np.exp(-np.linalg.norm(X.T[j] - W.T[i]))
        E = X - W @ H
        a = X.size * p / (E**2).sum()
        U = a * (1 - np.tanh(a * np.abs(E))**2)
        HD2 = (H * D)**2
        W *= (U * X @ H.T + 2 * y * X @ HD2.T) / (
            (U * (W @ H)) @ H.T + 2 * y * W * HD2.sum(axis=1))
        H *= W.T @ (U * X) / (W.T @ (U * (W @ H)) + b * H + y * H * D * D)
        error = rre(X, W, H)
        if (previous_error - error) / error_at_init < tol:
            break
        previous_error = error
    return W, H


def no_noise(shape: Tuple[int, int]) -> float:
    """No noise"""
    return 0


def salt_and_pepper(shape: Tuple[int, int],
                    p: float = 0.4,
                    r: float = 0.3) -> np.ndarray:
    """Randomly change some pixels to black or white"""
    p_noise = np.random.rand(*shape) <= p
    r_noise = np.random.rand(*shape) <= r
    return 1 * (p_noise * r_noise) - 1 * (p_noise * ~r_noise)


def uniform(shape: Tuple[int, int], scale: float = 0.4) -> np.ndarray:
    """Uniform noise"""
    return scale * (np.random.rand(*shape) - 0.5)


def laplace(shape: Tuple[int, int]) -> np.ndarray:
    """Laplace noise"""
    return np.random.laplace(0.5, 0.5)


def gaussian(shape: Tuple[int, int]) -> np.ndarray:
    """Gaussian noise"""
    return np.random.normal(0.5, 0.5)


def rre(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """Relative Reconstruction Error"""
    return np.linalg.norm(V - W @ H) / np.linalg.norm(V)


def evaluate_algorithm(
    V: np.ndarray, V_hat: np.ndarray, Y_hat: np.ndarray,
    algorithm: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                        np.ndarray]]
) -> None:
    """Fit model and run evaluation metrics"""
    print(f'{algorithm.__name__}: ', end='', flush=True)
    W, H = algorithm(len(set(Y_hat)), V)

    # Assign cluster labels.
    Y_pred = assign_cluster_label(H.T, Y_hat)

    acc = accuracy_score(Y_hat, Y_pred)
    nmi = normalized_mutual_info_score(Y_hat, Y_pred)
    print('RRE, Acc(NMI) = {:.4f}, {:.4f} ({:.4f})'.format(
        rre(V_hat, W, H), acc, nmi))


def main():
    """Run all algorithms"""
    for dataset, red, imgsize in ('ORL', 3, (92, 112)), ('CroppedYaleB', 4,
                                                         (168, 192)):
        # Load dataset.
        print(f'==> Load {dataset} dataset ...')
        V_hat, Y_hat = load_data(f'data/{dataset}', red)
        V_hat, _, Y_hat, _ = train_test_split(V_hat.T, Y_hat, train_size=0.9)
        V_hat = V_hat.T
        print(f'V_hat.shape={V_hat.shape}, Y_hat.shape={Y_hat.shape}')

        for noise in no_noise, salt_and_pepper, uniform, laplace, gaussian:
            # Add Noise
            print(f'==> Add {noise.__name__} noise ...')
            V = np.clip(V_hat + noise(V_hat.shape), 0, 1)

            for algorithm in zehu4485, ngra5777:
                evaluate_algorithm(V, V_hat, Y_hat, algorithm)


if __name__ == '__main__':
    main()
