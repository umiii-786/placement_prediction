from src.logging_config import logger
import pandas as pd 
import os 
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import mlflow

def load_data(path: str) -> tuple:
    try:
        logger.debug("Loading proceed train dataset")

        train_path = os.path.join(path, 'transformed_train_ds.csv')
        test_path = os.path.join(path, 'transformed_test_ds.csv')
        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)

        logger.debug(
            f"Datasets loaded successfully. "
            f"Train shape: {train_ds.shape} and Test shape: {train_ds.shape}"
        )

        return train_ds,test_ds

    except Exception as e:
        logger.error(f"Error occurred while loading datasets for testing: {e}")
        raise e
    
def load_model_and_label(model_path:str,label_path:str)->tuple:
    try:
        logger.debug("loadding model and label encoder")
        with open(model_path, 'rb') as f:
           model= pickle.load(f)

        with open(label_path, 'rb') as f:
           label=pickle.load(f)

        logger.debug("Model and label encoder loaded successfully")

        return model,label

    except Exception as e:
        logger.error(f"Error occurred while saving model and label encoder: {e}")
        raise e

  
def get_prediction(model, label, train_ds, test_ds):
    try:
        logger.debug("Starting model evaluation process")

        # Encode targets
        y_train = label.transform(train_ds['PlacementStatus'])
        y_test = label.transform(test_ds['PlacementStatus'])

        # Split features
        x_train = train_ds.drop('PlacementStatus', axis=1)
        x_test = test_ds.drop('PlacementStatus', axis=1)

        logger.debug(
            f"Evaluation data prepared. "
            f"Train shape: {x_train.shape}, Test shape: {x_test.shape}"
        )

        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        logger.debug("Predictions generated successfully")

        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)

        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        mlflow.set_experiment(experiment_name='rf')
        with mlflow.start_run():

            mlflow.log_metric("Training Accuracy", train_accuracy)
            mlflow.log_metric("Testing Accuracy", test_accuracy)

            mlflow.log_metric("Training Precision",train_precision)
            mlflow.log_metric("Testing Precision", test_precision)

            mlflow.log_metric("Training Recall", train_recall)
            mlflow.log_metric("Testing Recall",test_recall)

            mlflow.log_metric("Training F1 Score", train_f1)
            mlflow.log_metric("Testing F1 Score", test_f1)

    except Exception as e:
        logger.error(f"Error occurred during model evaluation: {e}")
        raise e


def main() -> None:
    try:
        logger.debug("Model evaluation pipeline started")

        load_data_path = os.path.join('data', 'processed')
        model_path = os.path.join('models', 'rf_model.pkl')
        label_path = os.path.join('models', 'rf_label.pkl')

        train_ds, test_ds = load_data(load_data_path)

        model, label = load_model_and_label(model_path, label_path)

        get_prediction(model, label, train_ds, test_ds)

        logger.debug("Model evaluation pipeline completed successfully")

    except Exception as e:
        logger.error(f"Model evaluation pipeline failed: {e}")
        raise e


if __name__ == '__main__':
    main()



