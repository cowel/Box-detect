import sys
from os import listdir
import numpy as np
import pickle
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from helper import manual
from config import AUTO_PRINT_HELP

default_image_size = tuple((45,45))

def get_image_matrix(image_dir):
    try:
        image_grayscale = Image.open(image_dir).convert('L')
        #resize image
        image_grayscale = image_grayscale.resize(default_image_size, Image.ANTIALIAS)
        #
        
        image_np = np.array(image_grayscale)
        img_list = []
        for line in image_np:
            for value in line:
                img_list.append(value)
        return img_list
    except Exception as e:
        return None

def get_train_test_images_from_directory(dataset_dir):
    X_train, X_test, Y_train, Y_test = [], [], [], []
    try:
        directory_list = listdir(dataset_dir)
        # remove '.DS_Store' from list
        if '.DS_Store' in directory_list:
            directory_list.remove('.DS_Store')
        # remove empty directory
        for directory in directory_list:
            if (len(f"{dataset_dir}/{directory}")) < 1 :
                directory_list.remove(directory)
                print("del: ", directory)
        # check for empty dataset folder
        if len(directory_list) < 1 :
            print("Train Dataset folder is empty or dataset folder contains no image")
            return None, None, None, None
        
        for directory in directory_list:
            print(directory)
            image_dir = listdir(f"{dataset_dir}/{directory}")
            if '.DS_Store' in image_dir:
                image_dir.remove('.DS_Store')
            
            split_point = int(0.8*len(image_dir))
            train_images, test_images = image_dir[:1000], image_dir[100:]

            for images in train_images:
                X_train.append(get_image_matrix(f"{dataset_dir}/{directory}/{images}"))
                Y_train.append(directory)
            for images in test_images:
                X_test.append(get_image_matrix(f"{dataset_dir}/{directory}/{images}"))
                Y_test.append(directory)

        return X_train, X_test, Y_train, Y_test

    except Exception as e:
        print(f"Error : {e}")
        return None, None, None, None

def train_model():
    train_dataset_dir = "./Dataset"
    X_train, X_test, Y_train, Y_test = get_train_test_images_from_directory(train_dataset_dir)
    if X_train is not None and X_test is not None and Y_train is not None and Y_test is not None :
        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(X_train,Y_train)
        accuracy_score = random_forest_classifier.score(X_train,Y_train)
        # save classifier
        pickle.dump(random_forest_classifier,open("Model/random_forest_classifier.pkl",'wb'))
        print(f"Model Accuracy Score : {accuracy_score}")
        test_accuracy_score = random_forest_classifier.score(X_test,Y_test)
        print(f"Model Accuracy Score (Test) : {test_accuracy_score}")
    else :
        print("An error occurred.")


def main():
    try:
        image_directory = '156.jpg'
        image_file = [get_image_matrix(image_directory)]
        print(image_file)
        # load saved model
        try:
            print("1111")
            with open("Model/random_forest_classifier.pkl",'rb') as input_model:
                saved_decision_tree_classifier_model = pickle.load(input_model)
            model_prediction = saved_decision_tree_classifier_model.predict(image_file)
            print(f"Recognized Digit : {model_prediction[0]} ")
            print("Day la so 0")
            # print(f"Recognized Digit : {model_prediction[0]} \nLatex Equivalent : {transform_to_latex(model_prediction[0])}")
        except FileNotFoundError as model_file_error:
            print(f"Error : {model_file_error}")
            print("... Training Model")
            train_model()
        
    except FileNotFoundError as file_error:
        print(f"Error : {file_error}")
    except Exception as e:
        print(f"Error : {e}")



args = sys.argv
if AUTO_PRINT_HELP and not('--no-manual in args'):
    print(manual.MANUAL)
print(args)
# main()
