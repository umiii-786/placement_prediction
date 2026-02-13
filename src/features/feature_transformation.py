from sklearn.compose import make_column_transformer,ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer,PowerTransformer,MinMaxScaler
from src.logging_config import logger
import os 
import pandas as pd 

def load_data(path: str) -> tuple:
    try:
        logger.debug("Loading raw train and test datasets")

        train_path = os.path.join(path, 'interim_train_ds.csv')
        test_path = os.path.join(path, 'interim_test_ds.csv')

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






def apply_transformation(train_ds: pd.DataFrame,
                         test_ds: pd.DataFrame) -> tuple:
    try:
        logger.debug("Starting feature transformation process")

        tf = ColumnTransformer(
            transformers=[
                ('onehot',
                 OneHotEncoder(drop='first', sparse_output=False),
                 [6, 7]),

                ('min max scaler',
                 MinMaxScaler(),
                 [0, 4, 5, 8, 9])
            ],
            remainder='passthrough'
        )

        logger.debug("Splitting features and target column")

        x_train = train_ds.drop('PlacementStatus', axis=1)
        x_test = test_ds.drop('PlacementStatus', axis=1)

        logger.debug("Fitting ColumnTransformer on training data")
        tf.fit(x_train)

        logger.debug("Transforming training and testing data")

        x_train_transformed = pd.DataFrame(
            tf.transform(x_train),
            columns=tf.get_feature_names_out()
        )

        x_test_transformed = pd.DataFrame(
            tf.transform(x_test),
            columns=tf.get_feature_names_out()
        )

        logger.debug(
            f"Transformation completed. "
            f"Train shape: {x_train_transformed.shape}, "
            f"Test shape: {x_test_transformed.shape}"
        )

        # Adding target column back
        x_train_transformed['PlacementStatus'] = train_ds['PlacementStatus'].values
        x_test_transformed['PlacementStatus'] = test_ds['PlacementStatus'].values

        logger.debug("Target column added back successfully")

        return x_train_transformed, x_test_transformed

    except Exception as e:
        logger.error(f"Error occurred during feature transformation: {e}")
        raise e


def save_data(train_ds: pd.DataFrame, test_ds: pd.DataFrame, path: str) -> None:
    try:
        logger.debug("Saving complete transformed datasets")

        os.makedirs(path, exist_ok=True)

        train_path = os.path.join(path, 'transformed_train_ds.csv')
        test_path = os.path.join(path, 'transformed_test_ds.csv')

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.debug(f"transformed datasets saved successfully at {path}")

    except Exception as e:
        logger.error(f"Error occurred while saving transformed  datasets: {e}")
        raise e


def main() -> None:
    try:
        logger.debug("Feature engineering pipeline started")

        load_path = os.path.join('data', 'interim')
        save_path = os.path.join('data', 'processed')

        train_ds, test_ds = load_data(load_path)

        transformed_train, transformed_test = apply_transformation(
            train_ds, test_ds
        )

        save_data(transformed_train, transformed_test, save_path)

        logger.debug("Feature engineering pipeline completed successfully")

    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()