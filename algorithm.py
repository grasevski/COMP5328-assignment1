#!/usr/bin/env python3
"""Run NMF algorithms and calculate evaluation metrics"""
from collections import Counter
from csv import DictWriter
import os
from pathlib import Path
import sys
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

IMAGE_PATH = 'figures'
SCALE = 255


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
            img = np.asarray(img).reshape((-1, 1)) / SCALE

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


def nmf_baseline(K: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Basic NMF algorithm, using sklearn library"""
    model = NMF(n_components=K)
    W = model.fit_transform(X)
    H = model.components_
    return W, H


def nmf(K: int,
        X: np.ndarray,
        beta: float = 2,
        l1: float = 0,
        l2: float = 0,
        weight: Callable[[np.ndarray, np.ndarray, np.ndarray],
                         np.ndarray] = lambda x, w, h: 1,
        steps: int = 100,
        tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Generic NMF algorithm using multiplicative updates"""
    avg = np.sqrt(X.mean() / K)
    W, H = avg * np.random.rand(len(X), K), avg * np.random.rand(K, len(X[0]))
    for _ in range(steps):
        W, H, done = mur(X, W, H, weight(X, W, H), beta, l1, l2, tol)
        if done:
            break
    return W, H


@njit(fastmath=True, parallel=True)
def mur(X: np.ndarray, W: np.ndarray, H: np.ndarray, U: np.ndarray,
        beta: float, l1: float, l2: float,
        tol: float) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Weighted multiplicative update rule"""
    WH = W @ H
    d_W = U * X * WH**(beta - 2) @ H.T / (U * WH**(beta - 1) @ H.T + l1 +
                                          l2 * W)
    e_W = np.linalg.norm(W * (1 - d_W))
    W *= d_W
    WH = W @ H
    d_H = W.T @ (U * X * WH**(beta - 2)) / (W.T @ (U * WH**(beta - 1)) + l1 +
                                            l2 * H)
    H *= d_H
    e_H = np.linalg.norm(H * (1 - d_H))
    return W, H, e_W < tol and e_H < tol


@njit(fastmath=True, parallel=True)
def err(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Difference between the original and the reconstructed"""
    return X - W @ H


@njit(fastmath=True, parallel=True)
def tanh_weight(X: np.ndarray,
                W: np.ndarray,
                H: np.ndarray,
                p: float = 1) -> np.ndarray:
    """Weight calculation for tanhNMF"""
    E = err(X, W, H)
    a = X.size * p / (E**2).sum()
    return a * (1 - np.tanh(a * np.abs(E))**2)


@njit(fastmath=True, parallel=True)
def cim_weight(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Weight calculation for CIM NMF"""
    E2 = err(X, W, H)**2
    return np.exp(-E2 / E2.mean())


@njit(fastmath=True)
def l1_weight(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Weight calculation for l1 NMF"""
    return 1 / np.linalg.norm(err(X, W, H), ord=1)


def l21_weight(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Weight calculation for l21 NMF"""
    return 1 / np.linalg.norm(err(X, W, H), axis=0)


def tanh_nmf(K: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Another Robust NMF"""
    return nmf(K, X, l2=1e-2, weight=tanh_weight)


def cim_nmf(K: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Robust NMF via half-quadratic minimization"""
    return nmf(K, X, weight=cim_weight)


def l21_nmf(K: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Robust nonnegative matrix factorization using l21-norm"""
    return nmf(K, X, weight=l21_weight)


def l1_nmf(K: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """“Non-negative Matrix Factorization for Images with Laplacian Noise"""
    return nmf(K, X, weight=l1_weight)


def kl_nmf(K: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Algorithms for nonnegative matrix factorization with the β-divergence"""
    return nmf(K, X, beta=1)


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


def uniform(shape: Tuple[int, int], scale: float = 0.1) -> np.ndarray:
    """Uniform noise"""
    return scale * (np.random.rand(*shape) - 0.5)


def laplace(shape: Tuple[int, int]) -> np.ndarray:
    """Laplace noise"""
    return np.random.laplace(0.5, 0.1)


def gaussian(shape: Tuple[int, int]) -> np.ndarray:
    """Gaussian noise"""
    return np.random.normal(0.5, 0.1)


def evaluate_algorithm(
    V: np.ndarray, V_hat: np.ndarray, Y_hat: np.ndarray,
    algorithm: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                        np.ndarray]]
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Fit model and run evaluation metrics"""
    W, H = algorithm(len(set(Y_hat)), V)

    # Assign cluster labels.
    Y_pred = assign_cluster_label(H.T, Y_hat)

    rre = np.linalg.norm(V - W @ H) / np.linalg.norm(V)
    acc = accuracy_score(Y_hat, Y_pred)
    nmi = normalized_mutual_info_score(Y_hat, Y_pred)
    return rre, acc, nmi, W, H


def run_nmf_algorithms(w: DictWriter, w_summary: DictWriter) -> None:
    """Run all combinations of algorithms and data and record results"""
    Y_hats = [None] * 5
    V_hats, Vs = Y_hats.copy(), Y_hats.copy()
    rre = np.zeros(len(Y_hats))
    acc, nmi = rre.copy(), rre.copy()
    np.random.seed(0)
    algorithms = [
        nmf_baseline, nmf, kl_nmf, l1_nmf, l21_nmf, cim_nmf, tanh_nmf
    ]

    for dataset, red, imgsize in ('ORL', 3, (92, 112)), ('CroppedYaleB', 4,
                                                         (168, 192)):
        # Load dataset.
        V_hat_orig, Y_hat_orig = load_data(f'data/{dataset}', red)
        img_size = [i // red for i in imgsize]

        for i in range(len(Y_hats)):
            V_hats[i], _, Y_hats[i], _ = train_test_split(V_hat_orig.T,
                                                          Y_hat_orig,
                                                          train_size=0.9)
            V_hats[i] = V_hats[i].T

        for noise in no_noise, salt_and_pepper, uniform, laplace, gaussian:
            # Add Noise
            Vs = [np.clip(v + noise(v.shape), 0, 1) for v in V_hats]

            plt.figure(figsize=(10, 3))
            ind = 2  # index of demo image

            row = {
                'dataset': dataset,
                'noise': noise.__name__.replace('_', '-'),
            }

            for i, v in enumerate(Vs):
                plt.subplot(len(Vs),
                            len(algorithms) + 1, (len(algorithms) + 1) * i + 1)
                if i == 0:
                    plt.title('input')
                plt.imshow(SCALE * v[:, ind].reshape(img_size[1], img_size[0]),
                           cmap=plt.cm.gray)
                plt.xticks(())
                plt.yticks(())

            for a, algorithm in enumerate(algorithms, 1):
                row['algorithm'] = algorithm.__name__.replace('_', '-')

                for i, (V, V_hat, Y_hat) in enumerate(zip(Vs, V_hats, Y_hats)):
                    rre[i], acc[i], nmi[i], W, H = evaluate_algorithm(
                        V, V_hat, Y_hat, algorithm)
                    plt.subplot(len(Vs),
                                len(algorithms) + 1,
                                (len(algorithms) + 1) * i + a + 1)
                    if i == 0:
                        plt.title(row['algorithm'])
                    plt.imshow(
                        SCALE *
                        (W @ H)[:, ind].reshape(img_size[1], img_size[0]),
                        cmap=plt.cm.gray)
                    plt.xticks(())
                    plt.yticks(())

                    w.writerow({
                        **row,
                        'trial': i + 1,
                        'RRE': rre[i],
                        'Acc': acc[i],
                        'NMI': nmi[i],
                    })

                w_summary.writerow({
                    **row,
                    'RRE': rre.mean(),
                    'RRE_std': rre.std(),
                    'Acc': acc.mean(),
                    'Acc_std': acc.std(),
                    'NMI': nmi.mean(),
                    'NMI_std': nmi.std(),
                })

            plt.savefig(f'{IMAGE_PATH}/{dataset}_{noise.__name__}.png')


def main() -> None:
    """Run all algorithms"""
    header = ['dataset', 'noise', 'algorithm', 'RRE', 'Acc', 'NMI']
    Path(IMAGE_PATH).mkdir(parents=True, exist_ok=True)
    w_summary = DictWriter(sys.stdout,
                           header + ['RRE_std', 'Acc_std', 'NMI_std'])
    w_summary.writeheader()
    with open('results.csv', 'w') as f:
        w = DictWriter(f, header + ['trial'])
        w.writeheader()
        run_nmf_algorithms(w, w_summary)


if __name__ == '__main__':
    main()
