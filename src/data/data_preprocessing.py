import pandas as pd
import os
from src.logging_config import logger  # assuming same logger config


def load_data(path: str) -> tuple:
    try:
        logger.debug("Loading raw train and test datasets")

        train_path = os.path.join(path, 'raw_train_ds.csv')
        test_path = os.path.join(path, 'raw_test_ds.csv')

        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)

        logger.debug(
            f"Datasets loaded successfully. "
            f"Train shape: {train_ds.shape}, Test shape: {test_ds.shape}"
        )

        return train_ds, test_ds

    except Exception as e:
        logger.error(f"Error occurred while loading datasets: {e}")
        raise e


def remove_duplicating(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Removing StudentID column and duplicates")

        df = df.copy()

        if 'StudentID' in df.columns:
            df.drop('StudentID', axis=1, inplace=True)
            logger.debug("StudentID column dropped successfully")

        before_rows = df.shape[0]
        df = df.drop_duplicates()
        after_rows = df.shape[0]

        logger.debug(
            f"Duplicates removed successfully. "
            f"Rows before: {before_rows}, Rows after: {after_rows}"
        )

        return df

    except Exception as e:
        logger.error(f"Error occurred while removing duplicates: {e}")
        raise e


def save_data(train_ds: pd.DataFrame, test_ds: pd.DataFrame, path: str) -> None:
    try:
        logger.debug("Saving interim datasets")

        os.makedirs(path, exist_ok=True)

        train_path = os.path.join(path, 'interim_train_ds.csv')
        test_path = os.path.join(path, 'interim_test_ds.csv')

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.debug(f"Interim datasets saved successfully at {path}")

    except Exception as e:
        logger.error(f"Error occurred while saving interim datasets: {e}")
        raise e


def main() -> None:
    try:
        logger.debug("Data cleaning pipeline started")

        load_path = os.path.join('data', 'raw')
        save_path = os.path.join('data', 'interim')

        train_ds, test_ds = load_data(load_path)

        train_ds = remove_duplicating(train_ds)

        save_data(train_ds, test_ds, save_path)

        logger.debug("Data cleaning pipeline completed successfully")

    except Exception as e:
        logger.error(f"Data cleaning pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()
