'''
This script parses face landmarks from a text file, normalizes the data using min-max normalization, 
and saves the processed data as a numpy array. The script can also be configured to use only the 
X and Y coordinates of the landmarks.
Functions:
    minmax_normalize(numbers_int: list) -> list:
    main(flags: argparse.Namespace):
Usage:
    Run the script with the following command-line arguments:
        --txt_path: Path to the input text file containing face landmarks data.
        --output_path: Path to save the output numpy array file.
        --minmax_nor: Whether to use minmax normalization (default: True).
        --xy_only: Whether to use only X and Y coordinates (default: True).
'''
import argparse
import numpy as np

MODEL_OUTPUT_SHAPE = 936

def minmax_normalize(numbers_int: list):
    """
    Normalize a list of integers using min-max normalization for 3D coordinates.
    This function takes a list of integers representing 3D coordinates (X, Y, Z) 
    and normalizes each coordinate independently using min-max normalization. 
    The normalization is performed separately for X, Y, and Z coordinates.
    Args:
        numbers_int (list): A list of integers representing 3D coordinates. 
                            The list should be structured such that every third 
                            element corresponds to the same coordinate axis 
                            (e.g., [X1, Y1, Z1, X2, Y2, Z2, ...]).
    Returns:
        list: A list of normalized integers where each coordinate axis has been 
              normalized independently.
    """
    def normalize_1d(ix, numbers_int):

        min_value = min(numbers_int[ix::3])
        max_value = max(numbers_int[ix::3])
        def normalize(x):
            return (x - min_value) / (max_value - min_value)
        numbers_int = [normalize(num) if idx % 3 == ix else num for idx, num in enumerate(numbers_int)]
        return numbers_int
    numbers_int = normalize_1d(0, numbers_int) # normalize X
    numbers_int = normalize_1d(1, numbers_int) # normalize Y
    numbers_int = normalize_1d(2, numbers_int) # normalize Z

    return numbers_int

def main(flags):
    """
    Main function to parse face landmarks from a text file, normalize the data, and save it as a numpy array.
    Args:
        flags (argparse.Namespace): A namespace object containing the following attributes:
            - txt_path (str): Path to the input text file containing face landmarks data.
            - output_path (str): Path to save the output numpy array file.
    Raises:
        ValueError: If the number of elements in a line does not match the expected MODEL_OUTPUT_SHAPE.
    """
    with open(flags.txt_path, 'r', encoding="utf-8") as f:
        content = f.readlines()

    landmarks_lines = [line for line in content if "INFO - Detected face landmarks:" not in line]

    landmarks = []
    for idx, line in enumerate(landmarks_lines):

        numbers = line.split(',')
        if len(numbers) == MODEL_OUTPUT_SHAPE:
            numbers_int = [ int(s.strip()) for s in  numbers]

            # minmax normalization
            if flags.minmax_nor:
                numbers_int = minmax_normalize(numbers_int)

            # remove the z axis
            if flags.xy_only:
                numbers_int = [num for idx, num in enumerate(numbers_int) if idx % 3 != 2]

            landmarks.append(numbers_int)
        else:
            print(f"{idx:{6}} index data format error which has {len(numbers)} elements!")
    landmarks = np.array(landmarks)
    print(f"The data shape: {landmarks.shape}")
    np.save(flags.output_path, landmarks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--txt_path',
        type=str,
        default = r'dataset\Normal_Face.txt',
        help='Path to txt file.')
    parser.add_argument(
        '--output_path',
        type=str,
        default = r'dataset\Normal_Face_XY_normal.npy',
        help='Path used for the output file.')
    parser.add_argument(
        '--minmax_nor',
        type=bool,
        default = True,
        help='Whether to use minmax normalization.')
    parser.add_argument(
        '--xy_only',
        type=bool,
        default = True,
        help='Whether to use only X and Y coordinates.')

    FLAGS, _ = parser.parse_known_args()
    main(FLAGS)
