from PIL import Image
import numpy as np
import glob

import tensorflow as tf


def get_dataset(mode="train", dataset=True):

    if (mode=="train"):
        dataset_path="datasets/training/"
    elif(mode == "valid"):
        dataset_path="datasets/validation/"
    elif(mode == "test"):
        dataset_path="datasets/test/"
    else:
        raise SystemExit(f'Not Supported Dataset Mode. Dataset Mode can be train|valid|test, but got {mode}')

    def read_images(filename):
        image = Image.open(filename)
        image =  np.array(image).astype("float32")

        #Preprocess
        image = image / 255.0
        return image

    def get_labels(filename):
        return int(filename[len(dataset_path)])

    filenames = glob.glob(f"{dataset_path}*.jpg")

    images = [read_images(filename) for filename in filenames]
    labels = [get_labels(filename)for filename in filenames]

    images = np.array(images).astype("float32")
    labels = np.array(labels).astype("float32")

    if not dataset:
        return images, labels, filenames

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    return dataset, len(images)