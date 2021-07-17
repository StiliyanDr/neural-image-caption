import copy
import json
import os
import random
from shutil import copy2 as copy_file

import tensorflow as tf
from tqdm import tqdm

from nic import utils


_MSCOCO_URL = "http://images.cocodataset.org"


def download_mscoco(directory=None, version="2017"):
    """
    Downloads the MS-COCO image captioning dataset.

    :param directory: a str - the directory where to download the data.
    If omitted, defaults to './mscoco' which is created if it does not
    exist. The dataset is overwritten if it already exists.
    :param version: a str - the year the data was published. Defaults to
    2017.
    """
    if (directory is None):
        directory = "mscoco"

        if (not os.path.exists(directory)):
            os.mkdir(directory)

    if (not os.path.exists(directory)):
        raise FileNotFoundError(f"'{directory}' does not exist!")

    _do_download_mscoco_to(directory, version)


def _do_download_mscoco_to(directory, version):
    _download_annotations(directory, version)
    _download_images(directory, f"train{version}")
    _download_images(directory, f"val{version}")


def _download_annotations(directory, version):
    annotations_dir = os.path.join(directory, "annotations")
    utils.remove_dir_if_exists(annotations_dir)

    annotation_zip = tf.keras.utils.get_file(
        "captions.zip",
        cache_subdir=os.path.abspath(directory),
        origin=f"{_MSCOCO_URL}/annotations/annotations_trainval{version}.zip",
        extract=True
    )
    os.remove(annotation_zip)


def _download_images(directory, zip_name):
    extracted_dir = os.path.join(directory, zip_name)
    utils.remove_dir_if_exists(extracted_dir)

    zip_name = f"{zip_name}.zip"
    images_zip = tf.keras.utils.get_file(
        zip_name,
        cache_subdir=os.path.abspath(directory),
        origin=f"{_MSCOCO_URL}/zips/{zip_name}",
        extract=True
    )
    os.remove(images_zip)


def split_out_test_data(directory="mscoco",
                        split=0.2,
                        version="2017",
                        verbose=True):
    """
    Randomly selects a set of train images (with random.shuffle) and
    moves them to a separate directory for testing purposes. Also
    extracts their captions to a separate file.

    :param directory: a str - the directory storing the mscoco dataset.
    Defaults to 'mscoco'.
    :param split: a float in the range (0, 1) - the percentage of train
    images to extract for testing. Defaults to `0.2` so 20%.
    :param version: a str - the dataset version. Defaults to "2017".

    Note that the train images are moved from 'train<version>' to a
    separate directory named 'test<version>' (overwritten if it exists).
    Their annotations are extracted from
    'annotations/captions_train<version>.json' to
    'annotations/captions_test<version>.json' but this extraction simply
    removes the annotations from the 'annotations' list in the first
    file and creates the second file which only contains the extracted
    annotations like so: `{"annotations": <annotations>}'.
    A copy of the original train captions file is created as back up.
    """
    train_dir = os.path.join(directory, f"train{version}")
    test_images = _select_test_images(train_dir, split)

    test_dir = os.path.join(directory, f"test{version}")
    utils.make_or_clear_dir(test_dir)
    _move_images(train_dir, test_dir, test_images, verbose)
    _extract_captions(directory, version, test_images, verbose)


def _select_test_images(directory, split):
    images = os.listdir(directory)
    random.shuffle(images)
    count = int(len(images) * split)

    return images[:count]


def _move_images(source, dest, images, verbose):
    if (verbose):
        print(f"Moving {len(images)} images "
              f"from '{source}' to '{dest}'.")
        images = tqdm(images)

    for i in images:
        os.rename(
            os.path.join(source, i),
            os.path.join(dest, i)
        )


def _extract_captions(directory, version, test_images, verbose):
    annotations_dir = os.path.join(directory, "annotations")
    train_captions_path = os.path.join(annotations_dir,
                                       f"captions_train{version}.json")
    copy_file(train_captions_path, f"{train_captions_path}.bkp")
    test_captions = _extract_test_captions_from(train_captions_path,
                                                test_images,
                                                verbose)
    test_captions_path = os.path.join(annotations_dir,
                                      f"captions_test{version}.json")
    _dump_to_file(test_captions, test_captions_path)


def _extract_test_captions_from(path, test_images, verbose):
    with open(path) as file:
        file_contents = json.load(file)

    test_image_ids = {utils.image_name_to_id(name)
                      for name in test_images}
    all_captions = file_contents["annotations"]
    test_captions = []
    reversed_indexes = range(len(all_captions) - 1, -1, -1)

    if (verbose):
        print(f"Extracting captions for {len(test_images)} test "
              f"images from {len(all_captions)} captions in '{path}'.")
        reversed_indexes = tqdm(reversed_indexes)

    for i in reversed_indexes:
        caption = all_captions[i]

        if (caption["image_id"] in test_image_ids):
            del all_captions[i]
            test_captions.append(caption)

    with open(path, "w") as file:
        json.dump(file_contents, file)

    return test_captions


def _dump_to_file(test_captions, path):
    with open(path, "w") as file:
        json.dump({"annotations": test_captions}, file)
