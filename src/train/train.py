import pickle

from cfg import (
    logger,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
)
from .test import test_model

def train_model(data_size, model, model_name, X, y, ampha=0.8):
    if not (X and y):
        logger.error('Dataset is empty')
        return

    train_size = min(round(ampha * len(X)), len(X) - 1)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logger.info('Training model %r with name %r', model.__class__.__name__, model_name)
    model.fit(X_train, y_train)

    logger.info('Test model with train set')
    test_model(model, X_train, y_train)
    logger.info('Test model with test set')
    test_model(model, X_test, y_test)

    # save classifier
    model_dir = f'{DEFAULT_MODEL_DIR}/{model_name}.pkl'
    logger.info('Saving model at: %r', model_dir)
    pickle.dump(model,open(model_dir,'wb'))
    logger.info('Model is saved')
