import sys
from os import listdir
import numpy as np
import pickle
import cv2
from sklearn.ensemble import RandomForestClassifier
from random import shuffle

from cfg import (
    logger,
    AUTO_PRINT_HELP,
    DEFAULT_MODEL_DIR,
)
from helper import manual
from train.test import test_model_by_image
from train.train import train_model


# def main(data_size=-1):
#     try:
        
#         # load saved model
#         try:
#             test_model(image_file)
#         except FileNotFoundError as model_file_error:
#             logger.info("... Training Model")
#             train_model(data_size)
#             logger.info("... Training Finishes")
#             test_model(image_file)
        
#     except FileNotFoundError as file_error:
#         logger.error("Error:  %r", file_error)
#     except Exception as e:
#         raise e


if __name__ == '__main__':
    args = sys.argv
    if AUTO_PRINT_HELP and not('--no-manual in args'):
        logger.info(manual.MANUAL)
    data_size = -1
    if '-l' in args:
        data_size = 10
        index_l = args.index('-l')
        if index_l + 2 <= len(args):
            try:
                data_size = int(args[index_l + 1])
            except Exception:
                pass
    if '-r' in args:
        train_model(data_size)
    logger.info('Data size: %r', data_size)
    # main(data_size)
    model_name = 'random_forest_classifier'
    image_dir = './Dataset/test/0.jpg'
    model_dir = f'{DEFAULT_MODEL_DIR}/{model_name}.pkl'
