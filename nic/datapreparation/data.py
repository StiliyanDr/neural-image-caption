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


def vocabulary_size(path):
    """
    :param path: a str - the path where preprocessed data is stored.
    :returns: an int - the size of the vocabulary obtained from train
    data.
    """
    tokenizer = load_tokenizer(path)
    return len(tokenizer.word_index)
