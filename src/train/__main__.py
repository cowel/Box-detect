import sys
from os import listdir
import numpy as np
import pickle
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from random import shuffle
from helper import manual
from config import (
    AUTO_PRINT_HELP,
    DEFAULT_IMAGE_SIZE,
    ACCEPTED_IMAGE_END
)


def get_image_matrix(image_dir):
    try:
        image_grayscale = Image.open(image_dir).convert('L')
        #resize image
        image_grayscale = image_grayscale.resize(DEFAULT_IMAGE_SIZE, Image.ANTIALIAS)
        #
        
        image_np = np.array(image_grayscale)
        img_list = []
        for line in image_np:
            for value in line:
                img_list.append(value)
        return img_list
    except Exception as e:
        return None


def get_train_test_images_from_directory(dataset_dir, data_size):
    X_train, X_test, Y_train, Y_test = [], [], [], []
    try:
        directory_list = listdir(dataset_dir)
        # print('directory_list:', directory_list)
        for directory in directory_list:
            if not (directory in '0123456789'):
                continue
            list_dir = listdir(f"{dataset_dir}/{directory}")
            if not list_dir:
                continue
            print('Make data from file:', f"{dataset_dir}/{directory}")
            image_dir = []
            for _dir in list_dir:
                if _dir.endswith(ACCEPTED_IMAGE_END):
                    image_dir.append(_dir)
            shuffle(image_dir)
            if data_size >= 0:
                image_dir = image_dir[:min(data_size, len(image_dir))]
            split_point = round(0.8*len(image_dir))
            print('split_point:', split_point)
            train_images, test_images = image_dir[:split_point], image_dir[split_point:]

            for images in train_images:
                X_train.append(get_image_matrix(f"{dataset_dir}/{directory}/{images}"))
                Y_train.append(directory)
            for images in test_images:
                X_test.append(get_image_matrix(f"{dataset_dir}/{directory}/{images}"))
                Y_test.append(directory)

        return X_train, X_test, Y_train, Y_test

    except Exception as e:
        raise e
        return None, None, None, None


def train_model(data_size):
    train_dataset_dir = "./Dataset"
    X_train, X_test, Y_train, Y_test = (
        get_train_test_images_from_directory(train_dataset_dir, data_size)
    )
    if X_train and X_test and Y_train and Y_test:
        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(X_train, Y_train)
        accuracy_score = random_forest_classifier.score(X_train, Y_train)
        print(f"Model Accuracy Score : {accuracy_score}")
        test_accuracy_score = random_forest_classifier.score(X_test, Y_test)
        print(f"Model Accuracy Score (Test) : {test_accuracy_score}")
        # save classifier
        pickle.dump(random_forest_classifier,open("Model/random_forest_classifier.pkl",'wb'))
    else :
        print("Data set it empty.")


def test_model(image_file):
    print("Load model")
    with open("Model/random_forest_classifier.pkl",'rb') as input_model:
        saved_model = pickle.load(input_model)
    predictions = saved_model.predict(image_file)
    print("Degit", 0, "had been predicted:", predictions[0])


def main(data_size=-1):
    try:
        image_directory = './Dataset/test/0.jpg'
        image_file = [get_image_matrix(image_directory)]
        # load saved model
        try:
            test_model(image_file)
        except FileNotFoundError as model_file_error:
            print("... Training Model")
            train_model(data_size)
            print("... Training Finishes")
            test_model(image_file)
        
    except FileNotFoundError as file_error:
        print(f"Error : {file_error}")
    except Exception as e:
        raise e


if __name__ == '__main__':
    args = sys.argv
    if AUTO_PRINT_HELP and not('--no-manual in args'):
        print(manual.MANUAL)
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
    print('Data size: ', data_size)
    main(data_size)
