import os
import torch
import OpenVisus as ov
from data_loader import VolumeDatasetLoader


if __name__ == '__main__':
    volume = 'lobster'
    dataset = VolumeDatasetLoader(volume)
    volume_dataset = ov.load_dataset(dataset.get_url(), cache_dir='./cache')
    data = volume_dataset.read(x=(0, dataset.get_xyz()[0]), y=(0, dataset.get_xyz()[1]), z=(0, dataset.get_xyz()[2]))

    print(data.shape)
    print(data.mean(axis=1).mean(axis=1))
