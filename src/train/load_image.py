from os import listdir
from random import shuffle
import numpy as np

from cfg import (
    logger,
    ACCEPTED_IMAGE_END
)
from helper.get_image import get_image_matrix


def unison_shuffled_copies(a, b):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p].tolist(), b[p].tolist()


def get_images_from_directory(dataset_dir, data_size):
    X, y = [], []
    try:
        directory_list = listdir(dataset_dir)
        for number in directory_list:
            # Only get folder with number
            if not (number in '0123456789'):
                continue
            list_dir = listdir(f"{dataset_dir}/{number}")
            # skip empty folder
            if not list_dir:
                continue
            logger.info('Build data from file: %r', f"{dataset_dir}/{number}")
            images_dir = []
            # Only get file with end in ACCEPTED_IMAGE_END list lik: .jpg
            for _dir in list_dir:
                if _dir.endswith(ACCEPTED_IMAGE_END):
                    images_dir.append(_dir)
            # shuffle(images_dir)
            if data_size >= 0:
                images_dir = images_dir[:min(data_size, len(images_dir))]

            for image_dir in images_dir:
                image = get_image_matrix(f"{dataset_dir}/{number}/{image_dir}")
                if image:
                    X.append(image)
                    y.append(number)
    except Exception as e:
        logger.error(e)

    return unison_shuffled_copies(X, y)
