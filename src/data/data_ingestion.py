import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
from src.logging_config import logger  # assuming you already created this


def load_data(dataset_path: str) -> pd.DataFrame:
    try:
        logger.debug("Loading dataset from KaggleHub")

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset_path,
            "placementdata.csv"
        )

        logger.debug(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error occurred while loading dataset: {e}")
        raise e


def load_yml(url: str) -> float:
    try:
        logger.debug("Loading YAML configuration file")

        with open(url, 'r') as file:
            content = yaml.safe_load(file)

        test_size = content['data_ingestion']['test_size']
        logger.debug(f"Test size loaded from YAML: {test_size}")

        return test_size

    except Exception as e:
        logger.error(f"Error occurred while loading YAML file: {e}")
        raise e


def split_data(df: pd.DataFrame, test_size: float) -> tuple:
    try:
        logger.debug("Splitting dataset into train and test")

        train_ds, test_ds = train_test_split(
            df,
            test_size=test_size,
            random_state=0,
            shuffle=True
        )

        logger.debug(
            f"Data split successful. Train shape: {train_ds.shape}, Test shape: {test_ds.shape}"
        )

        return train_ds, test_ds

    except Exception as e:
        logger.error(f"Error occurred while splitting data: {e}")
        raise e


def save_data(train_ds: pd.DataFrame, test_ds: pd.DataFrame, path: str) -> None:
    try:
        logger.debug("Saving train and test datasets")

        os.makedirs(path, exist_ok=True)

        train_path = os.path.join(path, 'raw_train_ds.csv')
        test_path = os.path.join(path, 'raw_test_ds.csv')

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.debug(f"Datasets saved successfully at {path}")

    except Exception as e:
        logger.error(f"Error occurred while saving datasets: {e}")
        raise e


def main() -> None:
    try:
        logger.debug("Data ingestion pipeline started")

        load_path = "ruchikakumbhar/placement-prediction-dataset"
        save_path = os.path.join('data', 'raw')

        df = load_data(load_path)
        test_size = load_yml('params.yaml')
        train_ds, test_ds = split_data(df, test_size)
        save_data(train_ds, test_ds, save_path)

        logger.debug("Data ingestion pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()
