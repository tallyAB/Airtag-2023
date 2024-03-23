import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-file', action='store')
parser.add_argument('-out', action='store')
args = parser.parse_args()

with open(args.file, 'r') as processed_file:
    lines = processed_file.readlines()
    mal_indices = []
    for i, line in enumerate(lines):
        if line.strip().endswith('+'):
            mal_indices.append(i)
    print(len(mal_indices))
    # Save ground truth indices as a NumPy array
    np.save(args.out, mal_indices)