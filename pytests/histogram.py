import os
import torch
import numpy as np
import OpenVisus as ov
import matplotlib.pyplot as plt

from data_loader import VolumeDatasetLoader


def find_peaks(volume, num_peaks, steepest=False):
    """
    Find peaks, rank them by their steepness.

    Given a parameter $num_peaks$ for the number of peaks to use for control points,
    return either the steepest n control points or the least steep n control points.
    """
    # Round
    rounded_data = np.round(volume)

    # Count
    bincount = np.bincount(rounded_data.flatten())

    # Slope
    slopes = [bincount[i] - bincount[i - 1] for i in range(1, len(bincount) - 2)]

    # Peaks
    peaks = [1 if slopes[i] > 0 and slopes[i + 1] < 0 else 0 for i in range(len(slopes) - 1)]
    peak_magnitudes = [0 if peaks[i] == 0 else abs(slopes[i]) + abs(slopes[i + 1]) for i in range(len(slopes) - 1)]

    # Sort
    idx = np.argsort(peak_magnitudes)[::-1]

    # Nonzero
    nonzero_elements_count = np.sum(np.where(np.array(peak_magnitudes) > 0, True, False))
    nonzero_idx = idx[:nonzero_elements_count]

    return nonzero_idx[:num_peaks]


if __name__ == '__main__':
    volume = 'tree'
    dataset = VolumeDatasetLoader(volume)
    volume_dataset = ov.load_dataset(dataset.get_url(), cache_dir='./cache')
    data = volume_dataset.read(x=(0, dataset.get_xyz()[0]), y=(0, dataset.get_xyz()[1]), z=(0, dataset.get_xyz()[2]))
    min_voxel, max_voxel = data.min(), data.max()

    # # Round
    # rounded_data = np.round(data)
    #
    # # Count
    # bincount = np.bincount(rounded_data.flatten())
    #
    # # Visualize
    # plt.stairs(bincount[1:])
    # plt.show()

    find_peaks(data, num_peaks=4, steepest=True)
