import numpy as np
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description='Load and print a NumPy array from a .npy file')
parser.add_argument('npy_file', type=str, help='Path to .npy file')

args = parser.parse_args()
array = np.load(args.npy_file)

print("Npy file:")
print(array)

