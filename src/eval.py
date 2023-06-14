import os, sys, argparse
import numpy as np


def calculate_dist(label, pred):
    assert label.shape[0] == pred.shape[0], 'The number of predicted results should be the same as the number of ground truth.'
    dist = np.sqrt(np.sum((label-pred)**2, axis=1))
    dist = np.mean(dist)
    return dist



def benchmark(dataset_path, sequeneces, src_path):
    if type(sequeneces) == str:
        sequeneces = [sequeneces]
    for seq in sequeneces:
        label = np.loadtxt(os.path.join(dataset_path, seq, 'gt_pose.txt'))
        pred = np.loadtxt(src_path, delimiter=" ")  #TODO: Enter your filename here#
        score = calculate_dist(label, pred)
        print(f'Mean Error of {seq}: {score:.5f}')
