"""
CNG403 Assignment 1 — Data Utilities
======================================
Shared utilities for data loading and feature extraction.
- Used by both reference.py and train.py
- HOG feature extraction with fixed parameters
"""

import random
import numpy as np
import torch
from skimage.feature import hog
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed) # Set seed for Python's built-in random module
    np.random.seed(seed) # Set seed for NumPy's random number generator
    torch.manual_seed(seed) # Set seed for PyTorch's random number generator (CPU)

#Generating random seed value for 1 library doesnt affect the other libraries,so we need to set seed for all libraries

# ---------------------------------------------------------------------------
# Feature extraction (HOG only)
#extract_hog_features takes the numpy arrays of the images and applies 
# ---------------------------------------------------------------------------

def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """
    Convert a batch of grayscale images (N, H, W) to HOG ( Histogram of Oriented Gradients ) feature vectors.

    Args:
        images: numpy array of shape (N, H, W), values in [0, 1].

    Returns:
        features: numpy array of shape (N, D).

    N-> number of images(samples)
    H-> height of the image
    W-> width of the image
    D-> dimension of the feature vector: the feature vector is a 1D array that contains 9 features since 
    """
    # Fixed HOG parameters
    # Each image is 28x28 pixels in the dataset initially, so don't need to resize the images for HOG
    # ?????? How can one be sure that the border pixels doesnt carry important information?
    # Orientation is the angle that we are looking for each pixel while we calculate the gradients w.r.t the neighbour pixels. 
    # 9 bins means that the degree of the angle in the bins is 0, 20, 40, 60, 80, 100, 120, 140, 160
    # The contribution of the pixels to the bins are weighted by the magnitude of the gradient, and according to the orientation of the gradient, the contribution is added to the corresponding bin.
    # Ex: If the magnitude of the gradient is 80 and the orientation is 25 degrees, the ratio of the distance of the orientation is 5/20 to 15/20 = 1/4 to 3/4 to the 20 and 40 degree bins, so we add the 1/4*80 to 20 degree bin and 3/4*80 to 40 degree bin.
    # We do this for all the pixels in the image (cell by cell????) and sum up the contributions to get the final feature vector for the image.
    # In the final we have 1x9 feature vector for each cell,

    HOG_ORIENTATIONS = 9 #20 degree bins
    HOG_PIXELS_PER_CELL = (8, 8) #We divide the image into 8x8 cells, so in total we have 3x3 cells and we ignore the remaining pixels
    HOG_CELLS_PER_BLOCK = (2, 2) #So the block is 

    features = []
    for img in images:
        feat = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            channel_axis=None,
        )
        features.append(feat)
    return np.stack(features)


def load_and_extract(data_root: str) -> tuple:
    """
    Download MNIST, extract HOG features, return torch tensors.

    Args:
        data_root: directory where MNIST should be stored/loaded.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test  — all torch.Tensor
    """
    # Fixed train/val split
    val_split = 0.1

#take the images from the MNIST dataset and transform to tensors(initially below code reads the images)
    to_tensor = transforms.ToTensor() #assigned variable for the transformation to tensor (better readability)
    train_raw = datasets.MNIST(data_root, train=True, download=True, transform=to_tensor) 
    test_raw = datasets.MNIST(data_root, train=False, download=True, transform=to_tensor)

#function to convert the dataset to numpy arrays and differentiate the images and labels
    def dataset_to_numpy(ds):
        imgs = ds.data.numpy().astype(np.float32) / 255.0  # (N, 28, 28)
        labels = ds.targets.numpy()
        return imgs, labels

#apply the above function to the train and test datasets (tensors) to convert them into arrays
    train_imgs, train_labels = dataset_to_numpy(train_raw)
    test_imgs, test_labels = dataset_to_numpy(test_raw)

    print("Extracting HOG features (train)…")
    X_train_full = extract_hog_features(train_imgs)
    print("Extracting HOG features (test)…")
    X_test = extract_hog_features(test_imgs)

    # Standardise using train statistics
    mean = X_train_full.mean(axis=0)
    std = X_train_full.std(axis=0) + 1e-8
    X_train_full = (X_train_full - mean) / std
    X_test = (X_test - mean) / std

    # Train / validation split
    n_val = int(len(X_train_full) * val_split)
    n_train = len(X_train_full) - n_val
    idx = np.random.permutation(len(X_train_full))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X_train_full[train_idx], train_labels[train_idx]
    X_val, y_val = X_train_full[val_idx], train_labels[val_idx]

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    to_l = lambda a: torch.tensor(a, dtype=torch.long)

    return to_t(X_train), to_l(y_train), to_t(X_val), to_l(y_val), to_t(X_test), to_l(test_labels)
