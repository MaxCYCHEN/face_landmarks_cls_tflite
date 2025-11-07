"""
This script performs inference using a TensorFlow Lite model on face landmark data to detect anomalies.
Functions:
    tflite_inference(input_data: np.ndarray, tflite_path: str) -> np.ndarray:
        Call forwards pass of TFLite file and returns the result.
    calculate_accuracy(predicted_indices, expected_indices) -> tf.Tensor:
        Calculates and returns accuracy.
    main(flags):
        Main function to load data, perform inference using TFLite model, and calculate accuracy.
    --tflite_path (str): Path to tflite file. Default is 'workspace/pose_anomaly_model_int8quant.tflite'.
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Model parameters
DATA_FEATURES_XYZ = 1404
DATA_FEATURES_XY = 936
MODEL_INPUT_SHAPE = DATA_FEATURES_XY
MODEL_CLASS_NUM = 2

NORMAL_DATA_PATH = os.path.join('dataset', 'Normal_Face_XY_normal.npy')
ANORMALY_DATA_PATH = os.path.join('dataset', 'Abnormal_Face_XY_normal.npy')

def tflite_inference(input_data: np.ndarray, tflite_path: str):
    """Call forwards pass of TFLite file and returns the result.

    Args:
        input_data (np.ndarray): Input data to use on forward pass.
        tflite_path (str): Path to TFLite file to run.

    Returns:
        Output from inference.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    # Check if the input/output type is quantized,
    # set scale and zero-point accordingly
    if input_dtype == np.int8:
        input_scale, input_zero_point = input_details[0]["quantization"]
    else:
        input_scale, input_zero_point = 1, 0

    input_data = input_data / input_scale + input_zero_point
    input_data = np.round(input_data) if input_dtype == np.int8 else input_data

    if output_dtype == np.int8:
        output_scale, output_zero_point = output_details[0]["quantization"]
    else:
        output_scale, output_zero_point = 1, 0

    interpreter.set_tensor(input_details[0]['index'], tf.cast(input_data, input_dtype))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

    return output_data

def calculate_accuracy(predicted_indices: list, expected_indices: list):
    """Calculates and returns accuracy.
    Args:
        predicted_indices (list): List of predicted integer indices.
        expected_indices (list): List of expected integer indices.
    Returns:
        Accuracy value between 0 and 1.
    """
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def main(flags: argparse.Namespace):
    """
    Main function to load data, preprocess it, and test a TensorFlow Lite model.
    Args:
        flags: An object containing the path to the TensorFlow Lite model file (tflite_path).
    """
    # load the data
    with open(NORMAL_DATA_PATH, 'rb') as f:
        normal_data = np.load(f)
    with open(ANORMALY_DATA_PATH, 'rb') as f:
        anomaly_data = np.load(f)
    train = np.concatenate((normal_data, anomaly_data), axis=0)

    normal_label = 0 * np.ones(normal_data.shape[0])
    anomaly_label = 1 * np.ones(anomaly_data.shape[0])
    labels = np.concatenate([normal_label, anomaly_label], axis =0)

    train_data, test_data, train_labels, test_labels = train_test_split(
        train, labels, test_size = 0.2, random_state = 29)

    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    # test tflite model
    tflite_path = flags.tflite_path

    predicted_indices = []

    for data, label in zip(test_data, test_labels):  # pylint: disable=unused-variable
        data = tf.cast(data, tf.float32) # Ensure data is float32

        print(data[0], data[1], data[2], data[3])

        prediction = tflite_inference(tf.expand_dims(data, axis=0), tflite_path)
        predicted_indices.append(np.squeeze(tf.argmax(prediction, axis=1)))

    test_accuracy = calculate_accuracy(predicted_indices, test_labels)
    confusion_matrix = tf.math.confusion_matrix(labels=tf.constant(test_labels),
                                                predictions=predicted_indices,
                                                num_classes = 2)
    print(confusion_matrix.numpy())
    print(f'Test accuracy = {test_accuracy * 100:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tflite_path', 
        '-t',
        type=str,
        default=r'workspace/XY_normalized/pose_anomaly_model_int8quant.tflite',
        help='Path to tflite file.')
    FLAGS, _ = parser.parse_known_args()
    main(FLAGS)
    