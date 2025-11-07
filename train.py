"""
This script trains a neural network model to classify normal and abnormal face landmarks.
Modules:
    os: Provides a way of using operating system dependent functionality.
    argparse: Parses command-line arguments.
    numpy: Supports large, multi-dimensional arrays and matrices.
    sklearn.model_selection: Provides train_test_split for splitting data into training and testing sets.
    tensorflow: An end-to-end open-source platform for machine learning.
    seaborn: A data visualization library based on matplotlib.
    matplotlib.pyplot: A state-based interface to matplotlib for plotting.
Constants:
    MODEL_NAME (str): The name of the model.
    WORKING_DIR (str): The working directory for saving the model and outputs.
    NORMAL_DATA_PATH (str): Path to the normal face landmarks data.
    ANORMALY_DATA_PATH (str): Path to the abnormal face landmarks data.
    DATA_FEATURES_XYZ (int): Number of features for XYZ data.
    DATA_FEATURES_XY (int): Number of features for XY data.
    MODEL_INPUT_SHAPE (int): Input shape for the model.
    MODEL_CLASS_NUM (int): Number of output classes for the model.
    EPOCHS (int): Number of epochs for training.
    BATCH_SIZE (int): Batch size for training.
    VALIDATION_SPLIT (float): Fraction of the training data to be used as validation data.
Classes:
    MyCallback: A custom callback to print the learning rate at the end of each epoch.
Functions:
    main(): The main function that loads data, trains the model, evaluates it, and converts it to TFLite format.
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# General parameters
MODEL_NAME = 'face_landmark_cls'
WORKING_DIR = "workspace"
NORMAL_DATA_PATH = os.path.join('dataset', 'Normal_Face_XY_normal.npy')
ANORMALY_DATA_PATH = os.path.join('dataset', 'Abnormal_Face_XY_normal.npy')

# Model parameters
DATA_FEATURES_XYZ = 1404
DATA_FEATURES_XY = 936
MODEL_INPUT_SHAPE = DATA_FEATURES_XY
MODEL_CLASS_NUM = 2

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.1

class MyCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to print the learning rate at the end of each epoch.
    Methods
    -------
    on_epoch_end(epoch, logs=None)
        Called at the end of each epoch. Prints the current learning rate.
    Parameters
    ----------
    epoch : int
        Index of the epoch.
    logs : dict, optional
        Dictionary of logs from the training process.
    """
    def on_epoch_end(self, epoch, logs=None):
        #lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        learning_rate = self.model.optimizer.learning_rate
        if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            # Calculate the current learning rate
            current_lr = learning_rate(self.model.optimizer.iterations)
        else:
            # Static learning rate
            current_lr =learning_rate
        print(f" lr for epoch {epoch + 1} is {current_lr.numpy():.8f}")

def main(flags):
    """
    Main function to load data, train a neural network model, evaluate it, and convert it to TensorFlow Lite format.
    Returns:
    None
    """

    # create the project folder
    proj_path = os.path.join(WORKING_DIR, flags.out_proj_name)
    try:
        os.makedirs(proj_path, exist_ok=True)
        print(f"Project folder '{proj_path}' created successfully.")
    except OSError as e:
        print(f"Error creating folder '{proj_path}': {e}")

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

    # train
    inputs = tf.keras.Input(shape=(MODEL_INPUT_SHAPE), name='input')
    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='random_uniform')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='random_uniform')(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform')(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='random_uniform')(x)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='random_uniform')(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    outputs = tf.keras.layers.Dense(MODEL_CLASS_NUM, activation='softmax', kernel_initializer='random_uniform')(x)
    model = tf.keras.Model(inputs, outputs)

    if flags.con_pw_decay:
        # A constant piecewise decay way to help learning. But the result is not as good as easy one.
        steps_num = (train_data.shape[1] * (1 - VALIDATION_SPLIT)) / BATCH_SIZE
        total_steps = EPOCHS * steps_num
        lr_boundary_list = [int(total_steps * 0.1), int(total_steps * 0.75)]
        learning_rates_list = [0.0005, 0.0002, 0.0001]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundary_list, values=learning_rates_list)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        # Easy way to set learning rate, but result is better.
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()

    if flags.con_pw_decay:
        # A constant piecewise decay way and steps_per_epoch
        model.fit(train_data, train_labels ,steps_per_epoch=steps_num,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                         callbacks=[MyCallback()])
    else:
        model.fit(train_data , train_labels , epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                     callbacks=[MyCallback()])

    # save the model
    tf.saved_model.save(model, os.path.join(proj_path, MODEL_NAME))

    # test result
    dnn_model = model  # pass the model in order to be used latter

    prediction = dnn_model.predict(test_data,verbose = 0)
    y_pred = np.argmax(prediction,axis = -1)

    conf_matrix = confusion_matrix(test_labels, y_pred)
    ax = sns.heatmap(conf_matrix, annot=True, fmt="d")
    ax.set(xlabel='Predicted Label', ylabel='True Label')
    # The top and bottom of heatmap gets trimmed off so to prevent that we set ylim
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(os.path.join(proj_path, "confusion_matrix.jpg"), dpi=200)

    pred = y_pred.astype(int)

    print(f"Accuracy = {accuracy_score(test_labels, pred)}")
    print(f"Precision = {precision_score(test_labels, pred)}")
    print(f"Recall = {recall_score(test_labels, pred)}")

    plt.title("Confusion matrix")
    plt.show()

    # convert the model to tflite
    def representative_dataset():
        for idx in range(200):
            data = train_data[idx]
            data = np.expand_dims(data, axis=0)
            yield [data.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    output_location = os.path.join(proj_path, (MODEL_NAME + r'_int8.tflite'))
    with open(output_location, 'wb') as f:
        f.write(tflite_model)
        print(f"The tflite output location: {output_location}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--con_pw_decay',
        action='store_true',
        help='Whether to use constant piecewise decay way and steps_per_epoch.')
    parser.add_argument(
        '--out_proj_name',
        '-o',
        type=str,
        default = r'XY_normalized',
        help='projet name in workspace.')

    FLAGS, _ = parser.parse_known_args()
    main(FLAGS)
