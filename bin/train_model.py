import sys
import numpy as np
import pickle
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from cfg import (
    logger,
    AUTO_PRINT_HELP,
    DEFAULT_MODEL_DIR,
    DEFAULT_DATASET_DIR,
)
from helper import manual
from train.test import test_model_by_image
from train.train import train_model
from train.load_image import get_images_from_directory
from train.model import MODELS, DEFAULT_MODEL


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
    logger.info('Data size: %r', data_size)

    default_model = DEFAULT_MODEL
    if '-m' in args:
        index_m = args.index('-m')
        if index_m + 2 <= len(args):
            try:
                default_model = int(args[index_m + 1])
            except Exception:
                pass
    logger.info('Data default_model: %r', MODELS[default_model][0].__class__.__name__)

    X, y = get_images_from_directory(DEFAULT_DATASET_DIR, data_size)
    if '-r' in args:
        if '-a' in args:
            for model in MODELS:
                train_model(data_size, model[0], model[1], X, y)
        else:
            model = MODELS[DEFAULT_MODEL][0]
            model_name = MODELS[DEFAULT_MODEL][1]
            train_model(data_size, model, model_name, X, y)

    model_name = 'random_forest_classifier'
    image_dir = './Dataset/test/0.jpg'
    model_dir = f'{DEFAULT_MODEL_DIR}/{model_name}.pkl'
