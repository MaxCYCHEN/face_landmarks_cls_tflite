"""
This module provides functionality to convert numpy arrays into 
C arrays and generate C header files.
Functions:
    create_array(np_array, var_type, var_name, array_dim=1, line_limit=80, indent=4):
        Converts a numpy array into a C array string.
    create_header(c_code, name):
        Creates a C header file string with the given C code and header guard.
    main(flags):
        Main function to load data, split it into training and testing sets, 
        convert test data into C arrays, and write them into a C header file.
Constants:
    NORMAL_DATA_PATH: Path to the normal face data numpy file.
    ANORMALY_DATA_PATH: Path to the abnormal face data numpy file.
Usage:
    Run the script with the desired output file name for the header file 
    using the --output_file_name argument.
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

NORMAL_DATA_PATH = os.path.join('dataset', 'Normal_Face_XY.npy')
ANORMALY_DATA_PATH = os.path.join('dataset', 'Abnormal_Face_XY.npy')

# Function to convert an array into a C string (requires Numpy)


def create_array(np_array, var_type, var_name, array_dim=1, line_limit=80, indent=4):
    """
    Converts a NumPy array into a C-style array declaration string.
    Args:
        np_array (numpy.ndarray): The NumPy array to be converted.
        var_type (str): The C variable type (e.g., 'float', 'int').
        var_name (str): The name of the C variable.
        array_dim (int, optional): The number of dimensions of the array. Defaults to 1.
        line_limit (int, optional): The maximum length of each line in the output string. Defaults to 80.
        indent (int, optional): The number of spaces to use for indentation. Defaults to 4.
    Returns:
        str: A string containing the C-style array declaration.
    """
    c_str = ''

    # Add array shape
    for i, dim in enumerate(np_array.shape):
        c_str += 'const unsigned int ' + var_name + \
            '_dim' + str(i + 1) + ' = ' + str(dim) + ';\n'
    c_str += '\n'

    # Declare C variable
    c_str += 'const ' + var_type + ' ' + var_name
    if array_dim == 1:  # 1 dim array
        one_dim_val = 1
        for dim in np_array.shape:
            one_dim_val = one_dim_val * dim
        c_str += '[' + str(one_dim_val) + ']'

    else:
        for dim in np_array.shape:
            c_str += '[' + str(dim) + ']'

    c_str += ' = {\n'

    # Create string for the array
    indent = ' ' * indent
    array_str = indent
    line_len = len(indent)
    val_sep = ', '
    for i, val in enumerate(np.nditer(np_array)):

        # Create a new line if string is over line limit
        val_str = str(val)
        if line_len + len(val_str) + len(val_sep) > line_limit:
            array_str += '\n' + indent
            line_len = len(indent)

        # Add value and separator
        array_str += val_str
        line_len += len(val_str)

        array_str += val_sep
        line_len += len(val_sep)

    # Add closing brace
    c_str += array_str + '\n};\n'

    return c_str


# Function to create a header file with given C code as a string
def create_header(c_code, name):
    """
    Generates a C header file content with header guards.
    Args:
        c_code (str): The C code to be included in the header file.
        name (str): The name to be used for the header guard.
    Returns:
        str: The complete content of the C header file with header guards.
    """
    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + name.upper() + '_H\n'
    c_str += '#define ' + name.upper() + '_H\n\n'

    # Add provided code
    c_str += c_code

    # Close out header guard
    c_str += '\n#endif //' + name.upper() + '_H'

    return c_str


def main(flags: argparse.Namespace):
    """
    Main function to load data, preprocess it, and write test data into a C header file.
    Args:
        flags (argparse.Namespace): An object containing the output file name and other configuration parameters.
    Raises:
        FileNotFoundError: If the data files specified by NORMAL_DATA_PATH or ANORMALY_DATA_PATH are not found.
        Exception: If there is an error during data loading, processing, or file writing.
    """
    # load the data
    with open(NORMAL_DATA_PATH, 'rb') as f:
        normal_data = np.load(f)
    with open(ANORMALY_DATA_PATH, 'rb') as f:
        anomaly_data = np.load(f)
    train = np.concatenate((normal_data, anomaly_data), axis=0)

    normal_label = 0 * np.ones(normal_data.shape[0])
    anomaly_label = 1 * np.ones(anomaly_data.shape[0])
    labels = np.concatenate([normal_label, anomaly_label], axis=0)

    train_data, test_data, train_labels, test_labels = train_test_split(
        train, labels, test_size=0.2, random_state=29)

    print(train_data.shape, test_data.shape,
          train_labels.shape, test_labels.shape)
    print(test_data[0])

    # write test data into C header file
    how_many_test = test_data.shape[0]

    x_test = test_data[:how_many_test, :]
    x_test_str = create_array(x_test, "uint8_t", "X_test")

    y_test = test_labels[:how_many_test].astype(int)
    y_test_str = create_array(y_test, "uint8_t", "y_test")

    test_d = x_test_str + y_test_str

    header_test_d = create_header(test_d, flags.output_file_name)
    with open(os.path.join("dataset", flags.output_file_name) + '.h', 'w', encoding='utf-8') as file:
        file.write(header_test_d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_file_name',
        type=str,
        default='facelandmarks_test_data',
        help='Name of the output header file.')

    FLAGS, _ = parser.parse_known_args()
    main(FLAGS)
