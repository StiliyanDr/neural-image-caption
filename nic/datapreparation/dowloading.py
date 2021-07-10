import os
import shutil

import tensorflow as tf


_MSCOCO_URL = "http://images.cocodataset.org"


def download_mscoco_to(directory=None, version="2017"):
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

    return _do_download_mscoco_to(directory, version)


def _do_download_mscoco_to(directory, version):
    annotations_dir = os.path.join(directory, "annotations")

    if (os.path.exists(annotations_dir)):
        shutil.rmtree(annotations_dir)

    annotation_file = _download_annotations(directory, version)

    images_dir = os.path.join(directory, f"train{version}")

    if (os.path.exists(images_dir)):
        shutil.rmtree(images_dir)

    images_dir = _download_images(directory, version)

    return (annotation_file, images_dir)


def _download_annotations(directory, version):
    annotation_zip = tf.keras.utils.get_file(
        "captions.zip",
        cache_subdir=os.path.abspath(directory),
        origin=f"{_MSCOCO_URL}/annotations/annotations_trainval{version}.zip",
        extract=True
    )
    annotation_file = os.path.join(os.path.dirname(annotation_zip),
                                   "annotations",
                                   f"captions_train{version}.json")
    os.remove(annotation_zip)

    return annotation_file


def _download_images(directory, version):
    images_zip = tf.keras.utils.get_file(
        f"train{version}.zip",
        cache_subdir=os.path.abspath(directory),
        origin=f"{_MSCOCO_URL}/zips/train{version}.zip",
        extract=True
    )
    images_dir = os.path.join(os.path.dirname(images_zip),
                              f"train{version}")
    os.remove(images_zip)

    return images_dir
