import os

import numpy as np
import tensorflow as tf

from nic.datapreparation import utils


def load_data(path, type, load_as_features=True):
    """
    :param path: a str - the path of the directory storing the
    preprocessed data.
    :param type: a str - the type of data to load. Possible values:
    'train', 'test' and 'val'.
    :param load_as_features: a boolean value indicating whether to
    load the image features. If `False`, the actual images (preprocessed
    for the chosen CNN) are loaded; this should be used only for fine
    tuning and testing. Defaults to `True`.
    :returns: a tf.data.Dataset which yields 3-tuples whose components
     are :
      - image tensors (feature vectors if `load_as_features` is set to
        `True`)
      - integer sequences (vectors) which represent the captions,
        without the end meta token at the end
      - integer sequences (vectors) which represent the captions,
        without the start meta token in front
    """
    data_subdir = os.path.join(path, type)
    captions = utils.deserialise_from(
        os.path.join(data_subdir, "captions.pcl")
    )
    images_dir = os.path.join(data_subdir,
                              ("features"
                               if (load_as_features)
                               else "images"))
    image_paths, all_captions = _vectorise(captions, images_dir, path)
    image_dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, all_captions)
    )

    return image_dataset.map(
        lambda path, caption:
        tf.numpy_function(
            _load_image,
            [path, caption],
            [tf.float32, tf.int32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )


def _vectorise(captions, images_dir, path):
    tokenizer = load_tokenizer(path)
    image_paths = []
    source_captions, target_captions = [], []

    for image_id, caps in captions.items():
        image_path = os.path.join(images_dir, f"{image_id}.pcl")
        image_paths.extend(image_path for _ in caps)

        caps = tokenizer.texts_to_sequences(caps)
        source_captions.extend(c[:-1] for c in caps)
        target_captions.extend(c[1:] for c in caps)

    source_captions = tf.keras.preprocessing.sequence.pad_sequences(
        source_captions,
        padding="post"
    )
    target_captions = tf.keras.preprocessing.sequence.pad_sequences(
        target_captions,
        padding="post"
    )
    all_captions = np.concatenate(
        [source_captions[:, np.newaxis, :],
         target_captions[:, np.newaxis, :]],
        axis=1
    )

    return (np.array(image_paths), all_captions)


def _load_image(path, caption):
    return (utils.deserialise_from(path.decode()).numpy(),
            caption[0, :],
            caption[1, :])


def load_tokenizer(path):
    """
    :param path: a str - the path where preprocessed data is stored.
    :returns: the tf.Tokenizer extracted from the train data.
    """
    tokenizer_path = os.path.join(path, "train", "tokenizer.json")

    with open(tokenizer_path) as file:
        contents = file.read()

    return tf.keras.preprocessing.text.tokenizer_from_json(
        contents
    )


def load_captions(path, type):
    """
    :param path: a str - the path where preprocessed data is stored.
    :param type: a str - the type of captions to load. Possible values:
    'train', 'test' and 'val'.
    :returns: a dictionary mapping image ids (int) to lists of captions
    (strs) which are surrounded by the start and end meta tokens.
    """
    return utils.deserialise_from(
        os.path.join(path, type, "captions.pcl")
    )


def load_images(path, type, load_as_features=False):
    """
    :param path: a str - the path where preprocessed data is stored.
    :param type: a str - the type of images to load. Possible values:
    'train', 'test' and 'val'.
    :param load_as_features: a boolean value indicating whether to
    load image features or just the preprocessed images. Defaults to
    `False`.
    :returns: a pair of a tf.data.Dataset which yields pairs of:
      - image tensors (feature vectors if `load_as_features` is set to
        `True`)
      - integers - the corresponding image ids
    and an int - the number of images in the Dataset.
    """
    data_subdir = os.path.join(path, type)
    images_dir = os.path.join(data_subdir,
                              ("features"
                               if (load_as_features)
                               else "images"))
    image_paths = [os.path.join(images_dir, name)
                   for name in os.listdir(images_dir)]
    image_dataset = tf.data.Dataset.from_tensor_slices(
        np.array(image_paths)
    )

    image_dataset = image_dataset.map(
        lambda path:
        tf.numpy_function(
            _do_load_image,
            [path],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return (image_dataset, len(image_paths))


def _do_load_image(path):
    path = path.decode()
    image = utils.deserialise_from(path).numpy()
    image_id = utils.image_name_to_id(utils.short_name_for(path))

    return (image, np.array(image_id, dtype=np.int32))


def vocabulary_size(path):
    """
    :param path: a str - the path where preprocessed data is stored.
    :returns: an int - the size of the vocabulary obtained from train
    data.
    """
    tokenizer = load_tokenizer(path)
    return len(tokenizer.word_index)


def features_size(path):
    """
    :param path: a str - the path where preprocessed data is stored.
    :returns: an int - the size of extracted image features.
    """
    features_dir = os.path.join(path, "train", "features")
    utils.verify_dir_exists(features_dir)
    all_features = os.listdir(features_dir)
    assert all_features
    features_path = os.path.join(features_dir, all_features[0])

    features = utils.deserialise_from(features_path)
    assert features.ndim == 1

    return int(tf.size(features))
