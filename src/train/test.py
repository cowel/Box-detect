import pickle

from cfg import (
    logger,
)
from helper.get_image import get_image_matrix


def test_model_by_image(model, image_dir, number):
    if isinstance(model, str):
        model_dir = model
        logger.info("Load model")
        with open(model_dir,'rb') as input_model:
            model = pickle.load(input_model)
    image = get_image_matrix(image_dir)
    predictions = model.predict(image)
    logger.info("Degit %d had been predicted: %r", number, predictions[0])


def test_model(model, X_test, y_test):
    test_accuracy_score = model.score(X_test, y_test)
    logger.warning("Model Accuracy Score %r:", test_accuracy_score)
