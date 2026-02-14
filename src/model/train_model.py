from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
import os
from src.logging_config import logger 
import pandas as pd 
import yaml
import pickle

def load_data(path: str) -> pd.DataFrame:
    try:
        logger.debug("Loading proceed train dataset")

        train_path = os.path.join(path, 'transformed_train_ds.csv')
        train_ds = pd.read_csv(train_path)

        logger.debug(
            f"Datasets loaded successfully. "
            f"Train shape: {train_ds.shape}"
        )

        return train_ds

    except Exception as e:
        logger.error(f"Error occurred while loading dataset for training: {e}")
        raise e

def load_parameters(url: str) -> float:
    try:
        logger.debug("Loading YAML configuration file for model Parameters")

        with open(url, 'r') as file:
            content = yaml.safe_load(file)

        parameters = content['lf_parameters']
        logger.debug(f"model Parameters loaded from YAML")

        return parameters

    except Exception as e:
        logger.error(f"Error occurred while loading YAML file for Parameters: {e}")
        raise e


def train_model(parameters: dict, x_train, y_train):
    try:
        logger.debug("Initializing RandomForestClassifier with parameters")

        model = RandomForestClassifier(**parameters)

        logger.debug("Starting model training")
        model.fit(x_train, y_train)

        logger.debug("Model training completed successfully")

        return model

    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        raise e


def save_model_and_label(model, lbl):
    try:
        logger.debug("Saving model and label encoder")

        os.makedirs('models', exist_ok=True)

        model_path = os.path.join('models', 'rf_model.pkl')
        label_path = os.path.join('models', 'rf_label.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(label_path, 'wb') as f:
            pickle.dump(lbl, f)

        logger.debug("Model and label encoder saved successfully")

    except Exception as e:
        logger.error(f"Error occurred while saving model and label encoder: {e}")
        raise e


def main() -> None:
    try:
        logger.debug("Model training pipeline started")

        load_path = os.path.join('data', 'processed')

        train_ds = load_data(load_path)
        parameters = load_parameters('params.yaml')

        logger.debug("Initializing LabelEncoder")
        lbl = LabelEncoder()

        logger.debug("Fitting LabelEncoder on target column")
        lbl.fit(train_ds['PlacementStatus'])

        x_train = train_ds.drop('PlacementStatus', axis=1)
        y_train = lbl.transform(train_ds['PlacementStatus'])

        logger.debug(
            f"Training data prepared. "
            f"Feature shape: {x_train.shape}, Target shape: {y_train.shape}"
        )

        model = train_model(parameters, x_train, y_train)

        save_model_and_label(model, lbl)

        logger.debug("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()
