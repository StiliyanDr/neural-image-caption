from collections import defaultdict
import itertools
import json
import os
import pickle
from tqdm import tqdm
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from nic.datapreparation import utils


class ImageOptions(NamedTuple):
    """
    Specifies image preprocessing options:
     - model_name: a str - the name of the model to preprocess the
       images for. This model is looked up in `tf.keras.applications`
       and its `preprocess_input` method is called on each image
     - target_size: a 2-tuple of integers - the spatial size of the
       image, as expected by the chosen model
     - feature_extractor: a callable taking and returning a `tf.Tensor`.
       If provided, this function will extract features for a batch of
       preprocessed images. This is useful when doing transfer
       learning and the feature extracting module of the model is frozen.
       Extracting the features once and reusing them to train the layers
       on top of it is more efficient
     - batch_size: an int - the batch size to use when preprocessing
       (and extracting features for) the images
    """
    model_name: str = "inception_resnet_v2"
    target_size: tuple = (299, 299)
    feature_extractor: object = None
    batch_size: int = 16


class MetaTokens(NamedTuple):
    """
    Bundles up meta tokens needed to represent sentences.
    """
    start: str = "<start>"
    end: str = "<end>"
    unknown: str = "<unk>"
    padding: str = "<pad>"


def preprocess_data(source_dir="mscoco",
                    target_dir="data",
                    version="2017",
                    image_options=ImageOptions(),
                    meta_tokens=MetaTokens(),
                    max_words=None,
                    verbose=True):
    """
    Preprocesses the MSCOCO dataset and stores the result on disk so
    that this can be done once and the result reused for each model
    training.

    Given the path to the directory where the dataset is stored and
    a path to the directory D where to store the preprocessed data,
    this function:
     - clears D or creates it if it does not exist
     - creates three subdirectories in D: train, test and val
     - preprocesses each image and pickles the `tf.Tensor` in
       'D/<train, test or val>/images/<image id>.pcl'. Similarly,
       features extracted for an image (if requested) are pickled to
       files (named the same way) in 'D/<train, test or val>/features'
     - loads the train captions, creates a
       `tf.keras.preprocessing.text.Tokenizer` for them and saves it
        to a file named 'D/train/tokenizer.json'
     - builds a dictionary mapping image ids (int) to a list of str
       captions (the original captions enclosed with the start and end
       meta tokens) and stores the mapping to a file named
       'D/<train, test or val>/captions.pcl'

    :param source_dir: a str - the directory storing the dataset.
    Defaults to 'mscoco'.
    :param target_dir: a str - the directory where to store the result.
    Defaults to 'data'.
    :param version: a str - the dataset's version. Defaults to '2017'.
    :param image_options: an instance of ImageOptions.
    :param meta_tokens: an instance of MetaTokens.
    :param max_words: the maximum vocabulary size. By default it is not
    limited.
    :param verbose: a boolean value indicating whether to show a status
    bar for the progress. Defaults to `True`.
    :raises FileNotFoundError: if the source directory does not exist.
    """
    utils.verify_dir_exists(source_dir)
    target_subdirs = _create_target_structure(target_dir)

    for data_type in ["val", "test", "train"]:
        source_images_dir = os.path.join(source_dir,
                                         f"{data_type}{version}")
        target_subdir = target_subdirs[data_type]
        preprocess_images(source_images_dir,
                          target_subdir,
                          image_options,
                          verbose)
        preprocess_captions(source_dir,
                            target_subdir,
                            meta_tokens,
                            data_type,
                            version,
                            max_words,
                            verbose)


def _create_target_structure(target_dir):
    utils.make_or_clear_dir(target_dir)
    train_dir = os.path.join(target_dir, "train")
    validation_dir = os.path.join(target_dir, "val")
    test_dir = os.path.join(target_dir, "test")
    utils.make_dirs([train_dir, validation_dir, test_dir])

    return {"train": train_dir, "val": validation_dir, "test": test_dir}


def preprocess_images(source_dir,
                      target_dir,
                      options=ImageOptions(),
                      verbose=True):
    """
    :param source_dir: a str - directory containing only image files.
    :param target_dir: a str - the directory where to store the
    preprocessed images (and optionally their features).
    :param options: an instance of ImageOptions.
    :param verbose: a boolean value indicating whether to show a status
    bar for the progress. Defaults to `True`.

    Note that each preprocessed image is a `tf.Tensor` which is pickled
    to a file in '<target_dir>/images' whose name is '<image id>.pcl'.
    Similarly, features extracted for an image (if requested) are stored
    as files (named the same way) in '<target_dir>/features'. If these
    two subdirectories exist, they are overwritten.
    """
    images_dir = os.path.join(target_dir, "images")
    utils.make_or_clear_dir(images_dir)

    features_dir = os.path.join(target_dir, "features")

    if (options.feature_extractor is not None):
        utils.make_or_clear_dir(features_dir)
    else:
        utils.remove_dir_if_exists(features_dir)

    images, images_count = _images_in(source_dir, options)
    model = getattr(tf.keras.applications, options.model_name)

    if (verbose):
        batches_count = _batches_count_for(images_count,
                                           options.batch_size)
        print(f"Preprocessing {images_count} images in '{source_dir}' "
              f"with a batch size of {options.batch_size}.")
        images = tqdm(images, total=batches_count)

    for images_batch, paths_batch in images:
        preprocessed_batch = model.preprocess_input(images_batch)
        features_batch = (options.feature_extractor(preprocessed_batch)
                          if (options.feature_extractor is not None)
                          else None)
        _serialise_batch(preprocessed_batch,
                         features_batch,
                         paths_batch,
                         features_dir,
                         images_dir)


def _images_in(directory, options):
    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        return (tf.image.resize(image, options.target_size), path)

    image_paths = [os.path.join(directory, name)
                   for name in os.listdir(directory)]
    image_dataset = tf.data.Dataset.from_tensor_slices(
        np.array(image_paths)
    )
    image_dataset = image_dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(options.batch_size)

    return (image_dataset, len(image_paths))


def _batches_count_for(total_items, batch_size):
    quotient, remainder = divmod(total_items, batch_size)
    return quotient + int(remainder != 0)


def _serialise_batch(images_batch,
                     features_batch,
                     paths_batch,
                     features_dir,
                     images_dir):
    features_batch = _yield_nones_if_missing(
        features_batch,
        count=images_batch.shape[0]
    )
    for image, features, path in zip(images_batch,
                                     features_batch,
                                     paths_batch):
        image_id = utils.image_name_to_id(
            utils.short_name_for(path.numpy().decode())
        )
        new_name = f"{image_id}.pcl"

        image_path = os.path.join(images_dir, new_name)
        utils.serialise(image, image_path)

        if (features is not None):
            features_path = os.path.join(features_dir, new_name)
            utils.serialise(features, features_path)


def _yield_nones_if_missing(iterable, count):
    return (iterable
            if (iterable is not None)
            else (None for _ in range(count)))


def preprocess_captions(source_dir,
                        target_dir,
                        meta_tokens=MetaTokens(),
                        type="train",
                        version="2017",
                        max_words=None,
                        verbose=True):
    """
    Loads the captions and builds a dictionary mapping image ids (int)
    to a list of str captions (the original captions enclosed with the
    start and end tokens) and stores the mapping on disk so that it can
    be reused. If type is 'train', a tf.keras.preprocessing.text.Tokenizer
    is built from the captions and is stored on disk.

    :param source_dir: a str - the directory where the mscoco dataset is
    stored.
    :param target_dir: a str - the directory where to save the mapping
    and tokenizer (in files named 'captions.pcl' and 'tokenizer.json',
    respectively).
    :param meta_tokens: an instance of MetaTokens - the meta tokens to
    use for the captions dictionary. Default ones can be used by
    omitting this argument.
    :param type: str - the type of captions to be loaded; 'val', 'test'
    or 'train'. Defaults to 'train'.
    :param version: a str - the captions' version. Defaults to '2017'.
    :param max_words: the maximum vocabulary size. By default it is not
    limited.
    :param verbose: a boolean value indicating whether to show a status
    bar for the progress. Defaults to `True`.
    """
    str_captions = _load_captions(source_dir,
                                  type,
                                  version,
                                  meta_tokens,
                                  verbose)

    if (type == "train"):
        all_captions = list(itertools.chain.from_iterable(
            str_captions.values()
        ))
        tokenizer = _create_tokenizer_for(all_captions,
                                          meta_tokens,
                                          max_words)
        _save_tokenizer(tokenizer, target_dir)

    _save_captions(str_captions, target_dir)


def _load_captions(data_dir, type, version, meta_tokens, verbose):
    """
    Returns a dictionary mapping image ids (int) to lists of captions
    (strs) which are surrounded by the start and end meta tokens.
    """
    file_name = f"captions_{type}{version}.json"
    path = os.path.join(data_dir, "annotations", file_name)

    with open(path) as file:
        contents = json.load(file)["annotations"]

    if (verbose):
        print(f"Loading {len(contents)} captions from '{path}'.")
        contents = tqdm(contents)

    captions = defaultdict(list)

    for item in contents:
        caption = " ".join([
            meta_tokens.start,
            item["caption"],
            meta_tokens.end,
        ])
        captions[item["image_id"]].append(caption)

    return captions


def _create_tokenizer_for(captions, meta_tokens, max_words=None):
    assert isinstance(captions, list)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=max_words,
        filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~',
        oov_token=meta_tokens.unknown
    )
    tokenizer.fit_on_texts(captions)
    tokenizer.word_index[meta_tokens.padding] = 0
    tokenizer.index_word[0] = meta_tokens.padding

    return tokenizer


def _save_tokenizer(tokenizer, target_dir):
    tokenizer_path = os.path.join(target_dir, "tokenizer.json")

    with open(tokenizer_path, "w") as file:
        file.write(tokenizer.to_json())


def _save_captions(caps, target_dir):
    captions_path = os.path.join(target_dir, "captions.pcl")

    with open(captions_path, "wb") as file:
        pickle.dump(caps, file)
