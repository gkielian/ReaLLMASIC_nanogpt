import numpy as np
import random
import argparse
import csv

def generate_letter_mapping(degrees):
    radians = np.deg2rad(degrees)
    return {
        'H': (np.cos(radians), np.sin(radians)),
        'M': (1.0, 0.0),
        'L': (np.cos(radians), -np.sin(radians)),
        'y': (np.cos(radians), np.sin(radians)),
        'n': (np.cos(radians), -np.sin(radians)),
        's': (np.cos(radians), np.sin(radians)),
        'a': (1.0, 0.0),
        'f': (np.cos(radians), -np.sin(radians)),
    }

# Generate random coordinates
def random_coordinates(mean=0.0, stdev=0.02):
    return (random.gauss(mean, stdev), random.gauss(mean, stdev))
# Map letters based on the mapping or generate random coordinates

def map_letter(letter, letter_mapping):
    if letter in letter_mapping:
        return letter_mapping[letter]
    elif letter == 'r':
        return random_coordinates()
    else:
        return random_coordinates()
# Scale numeric values to a unit circle

def map_numeric(value, min_value, max_value, max_angle_difference=180):
    scaled_value = (value - min_value) / (max_value - min_value)
    radians = np.pi * (max_angle_difference / 180.0) * scaled_value
    return np.cos(radians), np.sin(radians)

# Parse arguments
parser = argparse.ArgumentParser(description='Generate initial_wte.npy from a CSV file.')
parser.add_argument('--csv', type=str, required=True, help='Path to the input CSV file.')
parser.add_argument('--mode', type=str, choices=['letters', 'numeric'], default='letters', help='Mode: "letters" for letter mapping, "numeric" for numeric mapping.')
parser.add_argument('--degrees', type=int, default=60, help='Degrees of separation for letters (default: 60)')
parser.add_argument('--min', type=float, default=0.0, help='Minimum value for numeric scaling (default: 0.0)')
parser.add_argument('--max', type=float, default=1.0, help='Maximum value for numeric scaling (default: 1.0)')
parser.add_argument('--max_angle_difference', type=float, default=180.0, help='Maximum value difference of extremes.')
args = parser.parse_args()

# Load the CSV file
with open(args.csv, newline='') as csvfile:
    reader = csv.reader(csvfile)
    table = list(reader)

# Generate letter mapping
letter_mapping = generate_letter_mapping(args.degrees)

# Map the table based on the selected mode
if args.mode == 'letters':
    mapped_table = [[coord for letter in row for coord in map_letter(letter, letter_mapping)] for row in table]
elif args.mode == 'numeric':
    numeric_table = np.array(table, dtype=float)
    min_value = args.min
    max_value = args.max
    max_angle_difference = args.max_angle_difference
    mapped_table = [[x for value in row for x in map_numeric(value, min_value, max_value, max_angle_difference)] for row in numeric_table]

# Convert to numpy array
wte = np.array(mapped_table)
# Print the shape of the wte
print(f"Shape of wte: {wte.shape}")
# Save the wte as a .npy file
np.save('initial_wte.npy', wte)
print(f"Saved initial wte with shape {wte.shape} to initial_wte.npy")
# Print the first few rows of the wte with 3 decimal places
print("\nPrint wte (3 decimal places):")
np.set_printoptions(precision=3, suppress=True)
print(wte)
