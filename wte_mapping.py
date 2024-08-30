import numpy as np
import random
import argparse
import csv
from typing import Dict, Tuple, List

def generate_letter_mapping(degrees: int) -> Dict[str, Tuple[float, float]]:
    radians = np.deg2rad(degrees)
    cos, sin = np.cos(radians), np.sin(radians)
    return {
        'H': (cos, sin),
        'M': (1.0, 0.0),
        'L': (cos, -sin),
        'y': (cos, sin),
        'n': (cos, -sin),
        's': (cos, sin),
        'a': (1.0, 0.0),
        'f': (cos, -sin),
    }

def random_coordinates(mean: float = 0.0, stdev: float = 0.02) -> Tuple[float, float]:
    return random.gauss(mean, stdev), random.gauss(mean, stdev)

def map_letter(letter: str, letter_mapping: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    return letter_mapping.get(letter, random_coordinates())

def map_numeric(value: float, min_value: float, max_value: float, max_angle_difference: float = 180) -> Tuple[float, float]:
    scaled_value = (value - min_value) / (max_value - min_value)
    radians = np.pi * (max_angle_difference / 180.0) * scaled_value
    return np.cos(radians), np.sin(radians)

def load_csv(file_path: str) -> List[List[str]]:
    with open(file_path, newline='') as csvfile:
        return list(csv.reader(csvfile))

def map_table(table: List[List[str]], mode: str, letter_mapping: Dict[str, Tuple[float, float]],
              min_value: float, max_value: float, max_angle_difference: float) -> np.ndarray:
    if mode == 'letters':
        return np.array([[coord for letter in row for coord in map_letter(letter, letter_mapping)] for row in table])
    elif mode == 'numeric':
        numeric_table = np.array(table, dtype=float)
        return np.array([[x for value in row for x in map_numeric(value, min_value, max_value, max_angle_difference)]
                         for row in numeric_table])

def main():
    parser = argparse.ArgumentParser(description='Generate initial_wte.npy from a CSV file.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--mode', type=str, choices=['letters', 'numeric'], default='letters',
                        help='Mode: "letters" for letter mapping, "numeric" for numeric mapping.')
    parser.add_argument('--degrees', type=int, default=60, help='Degrees of separation for letters (default: 60)')
    parser.add_argument('--min', type=float, default=0.0, help='Minimum value for numeric scaling (default: 0.0)')
    parser.add_argument('--max', type=float, default=1.0, help='Maximum value for numeric scaling (default: 1.0)')
    parser.add_argument('--max_angle_difference', type=float, default=180.0,
                        help='Maximum value difference of extremes.')
    args = parser.parse_args()

    table = load_csv(args.csv)
    letter_mapping = generate_letter_mapping(args.degrees)

    wte = map_table(table, args.mode, letter_mapping, args.min, args.max, args.max_angle_difference)

    print(f"Shape of wte: {wte.shape}")
    np.save('initial_wte.npy', wte)
    print(f"Saved initial wte with shape {wte.shape} to initial_wte.npy")

    np.set_printoptions(precision=3, suppress=True)
    print("\nPrint wte (3 decimal places):")
    print(wte)

if __name__ == "__main__":
    main()
