from collections import defaultdict
import json
import os
import pickle
from typing import NamedTuple

import tensorflow as tf

from nic import utils


class ImageOptions(NamedTuple):
    """
    Specifies image preprocessing options:
     - model_name: a str - the name of the model to preprocess the
       images for. This model is looked up in `tf.keras.applications`
       and its `preprocess_input` method is called on each image
     - target_size: a 2-tuple of integers - the spatial size of the
       image, as expected by the chosen model
     - feature_extractor: a callable taking and returning a `tf.Tensor`.
       If provided, this function will extract features for each of
       the preprocessed images. This is useful when doing transfer
       learning and the feature extracting module of the model is frozen.
       Extracting the features once and reusing them to train the layers
       on top of it is more efficient
    """
    model_name: str = "inception_resnet_v2"
    target_size: tuple = (299, 299)
    feature_extractor: object = None


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
                    max_words=None):
    """
    Preprocesses the MSCOCO dataset and stores the result on disk so
    that this can be done once and the result reused for each model
    training.

    Given the path to the directory where the dataset is stored and
    a path to the directory D where to store the preprocessed data,
    this function:
     - clears D or creates it if it does not exist
     - creates two subdirectories in D: train and val
     - preprocesses each image and pickles the `tf.Tensor` in
       'D/<train or val>/images/<image id>.pcl'. Similarly, features
       extracted for an image (if requested) are pickled to files (named
       the same way) in 'D/<train or val>/features'
     - loads the captions, creates a tf.keras.preprocessing.text.Tokenizer
       for them, builds a dictionary mapping image ids (int) to a list
       of int sequences (the word indices) and saves the tokenizer and
       mapping to files in 'D/<train or val>' - 'captions.pcl' and
       'tokenizer.json', respectively.

    :param source_dir: a str - the directory storing the dataset.
    :param target_dir: a str - the directory where to store the result.
    :param version: a str - the dataset's version. Defaults to '2017'.
    :param image_options: an instance of ImageOptions.
    :param meta_tokens: an instance of MetaTokens.
    :param max_words: the maximum size of the captions dictionary. By
    default it is not limited.
    :raises FileNotFoundError: if the source directory does not exist.
    """
    utils.verify_dir_exists(source_dir)
    target_subdirs = _create_target_structure(target_dir)

    for data_type in ["val", "train"]:
        source_images_dir = os.path.join(source_dir,
                                         f"{data_type}{version}")
        target_subdir = target_subdirs[data_type]
        preprocess_images(source_images_dir,
                          target_subdir,
                          image_options)
        preprocess_captions(source_dir,
                            target_subdir,
                            meta_tokens,
                            data_type,
                            version,
                            max_words)


def _create_target_structure(target_dir):
    utils.make_or_clear_dir(target_dir)
    train_dir = os.path.join(target_dir, "train")
    validation_dir = os.path.join(target_dir, "val")
    utils.make_dirs([train_dir, validation_dir])

    return {"train": train_dir, "val": validation_dir}


def preprocess_images(source_dir,
                      target_dir,
                      options=ImageOptions()):
    """
    :param source_dir: a str - directory containing only image files.
    :param target_dir: a str - the directory where to store the
    preprocessed images (and optionally their features).
    :param options: an instance of ImageOptions.

    Note that each preprocessed image is a `tf.Tensor` which is pickled
    to a file in '<target_dir>/images' whose name is '<image id>.pcl'.
    Similarly, features extracted for an image (if requested) are stored
    as files (named the same way) in '<target_dir>/features'. If these
    two subdirectories exist, they are overwritten.
    """
    images_dir = os.path.join(target_dir, "images")
    utils.make_or_clear_dir(images_dir)

    features_dir = os.path.join(target_dir, "features")
    feature_extraction_requested = options.feature_extractor is not None

    if (feature_extraction_requested):
        utils.make_or_clear_dir(features_dir)
    else:
        utils.remove_dir_if_exists(features_dir)

    for path, image_path, features_path in _images_in(
        source_dir,
        images_dir,
        features_dir,
    ):
        image = _preprocess_image(path, options)
        utils.serialise(image, image_path)

        if (feature_extraction_requested):
            features = options.feature_extractor(image)
            utils.serialise(features, features_path)


def _images_in(directory, images_dir, features_dir):
    for name in os.listdir(directory):
        path = os.path.join(directory, name)

        end = name.rfind(".")
        image_id = name[:end].lstrip("0")
        new_name = f"{image_id}.pcl"
        image_path = os.path.join(images_dir, new_name)
        features_path = os.path.join(features_dir, new_name)

        yield (path, image_path, features_path)


def _preprocess_image(path, options):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, options.target_size)
    model = getattr(tf.keras.applications, options.model_name)

    return model.preprocess_input(image)


def preprocess_captions(source_dir,
                        target_dir,
                        meta_tokens=MetaTokens(),
                        type="train",
                        version="2017",
                        max_words=None):
    """
    Loads the captions and if they are:
     - training: creates a tf.keras.preprocessing.text.Tokenizer
       for them, builds a dictionary mapping image ids (int) to a list
       of int sequences (the word indices) and stores the tokenizer and
       mapping on disk so that they can be reused for training
    - validation: builds a dictionary mapping image ids (int) to a list
       of str captions (the original captions enclosed with the start
       and end tokens) and stores the mapping on disk so that it can be
       reused for validation

    :param source_dir: a str - the directory where the mscoco dataset is
    stored.
    :param target_dir: a str - the directory where to save the mapping
    and tokenizer (in files named 'captions.pcl' and 'tokenizer.json',
    respectively).
    :param meta_tokens: an instance of MetaTokens - the meta tokens to
    use for the captions dictionary. Default ones can be used by
    omitting this argument.
    :param type: str - the type of captions to be loaded; 'val' or
    'train'. Defaults to 'train'.
    :param version: a str - the captions' version. Defaults to '2017'.
    :param max_words: the maximum size of the captions dictionary. By
    default it is not limited.
    """
    str_captions = _load_captions(source_dir,
                                  type,
                                  version,
                                  meta_tokens)

    if (type == "train"):
        tokenizer = _create_tokenizer_for(str_captions,
                                          meta_tokens,
                                          max_words)
        int_captions = _vectorize_captions(str_captions, tokenizer)
        _save_captions(int_captions, target_dir)
        _save_tokenizer(tokenizer, target_dir)
    else:
        _save_captions(str_captions, target_dir)


def _load_captions(data_dir, type, version, meta_tokens):
    """
    Returns a dictionary mapping image ids (int) to lists of captions
    (strs) which are surrounded by the start and end meta tokens.
    """
    file_name = f"captions_{type}{version}.json"
    path = os.path.join(data_dir, "annotations", file_name)

    with open(path) as file:
        contents = json.load(file)["annotations"]

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
        oov_token=meta_tokens.unknown
    )
    tokenizer.fit_on_texts(captions)
    tokenizer.word_index[meta_tokens.padding] = 0
    tokenizer.index_word[0] = meta_tokens.padding

    return tokenizer


def _vectorize_captions(captions, tokenizer):
    return {
        image_id: tokenizer.texts_to_sequences(caps)
        for image_id, caps in captions.items()
    }


def _save_tokenizer(tokenizer, target_dir):
    tokenizer_path = os.path.join(target_dir, "tokenizer.json")

    with open(tokenizer_path, "w") as file:
        file.write(tokenizer.to_json())


def _save_captions(caps, target_dir):
    captions_path = os.path.join(target_dir, "captions.pcl")

    with open(captions_path, "wb") as file:
        pickle.dump(caps, file)
