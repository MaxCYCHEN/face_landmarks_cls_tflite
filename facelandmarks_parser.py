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
'''
import argparse
import numpy as np

MODEL_OUTPUT_SHAPE = 936

class ParserFlagsException(Exception):
    """
    Exception raised for errors in the parser flags.
    """
    def __init__(self, message: str):
        super().__init__(message)

def minmax_normalize(numbers_int: list):
    """
    Normalize a list of integers using min-max normalization.
    This function takes a list of integers and normalizes the values in the list
    such that the minimum value becomes 0 and the maximum value becomes 1. The 
    normalization is performed separately for the even-indexed and odd-indexed 
    elements in the list.
    Args:
        numbers_int (list): A list of integers to be normalized.
    Returns:
        list: A list of normalized integers.
    """
    def normalize_1d(ix, numbers_int):

        min_value = min(numbers_int[ix::2])
        max_value = max(numbers_int[ix::2])
        def normalize(x):
            return (x - min_value) / (max_value - min_value)
        numbers_int = [normalize(num) if idx % 2 == ix else num for idx, num in enumerate(numbers_int)]
        return numbers_int
    numbers_int = normalize_1d(0, numbers_int) # normalize X
    numbers_int = normalize_1d(1, numbers_int) # normalize Y

    return numbers_int

def forehead_relatively_normalize(numbers_int: list):
    """
    Normalize a list of integers relative to the forehead coordinates.
    Args:
        numbers_int (list): A list of integers to be normalized.
    Returns:
        list: A list of normalized integers.
    """

    forehead_x = numbers_int[0]
    forehead_y = numbers_int[1]
    img_scale_x = 192.0
    img_scale_y = 192.0

    numbers_int = [(num - forehead_x)/img_scale_x if idx % 2 == 0 else (num - forehead_y)/img_scale_y for idx, num in enumerate(numbers_int)]

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
            if flags.minmax_norm:
                numbers_int = minmax_normalize(numbers_int)

            # forehead relatively normalization
            if flags.forehead_norm:
                numbers_int = forehead_relatively_normalize(numbers_int)

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
        '-t',
        type=str,
        default = r'dataset\Normal_Face.txt',
        help='Path to txt file.')
    parser.add_argument(
        '--output_path',
        '-o',
        type=str,
        default = r'dataset\Normal_Face_XY_normal.npy',
        help='Path used for the output file.')
    parser.add_argument(
        '--minmax_norm',
        action='store_true',
        help='Whether to use minmax normalization.')
    parser.add_argument(
        '--forehead_norm',
        action='store_true',
        help='Whether to use forehead relatively normalization.')

    FLAGS, _ = parser.parse_known_args()

    if FLAGS.minmax_norm and FLAGS.forehead_norm:
        raise ParserFlagsException("Both minmax_norm and forehead_norm flags are set! Please set only one of them.")

    main(FLAGS)
