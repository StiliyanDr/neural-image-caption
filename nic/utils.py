import os
import pickle
import shutil


def default_if_none(value, default):
    return (value
            if (value is not None)
            else default)


def serialise(obj, path):
    """
    Serialises a pickleable object to a file.

    :param obj: the object to serialise.
    :param path: a str - the path to the file.
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def deserialise_from(path):
    """
    Deserialises a pickleable object from a file.

    :param path: a str - the path of the file.
    :returns: the deserialised Python object.
    """
    with open(path, "rb") as file:
        return pickle.load(file)


def remove_dir_if_exists(directory):
    if (os.path.exists(directory)):
        shutil.rmtree(directory)
