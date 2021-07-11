import os

import tensorflow as tf

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
