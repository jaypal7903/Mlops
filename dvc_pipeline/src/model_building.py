import os
import pickle
import logging
import yaml
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# Project Paths
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# Logger Configuration
# =============================================================================

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "model_building.log")
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =============================================================================
# Functions
# =============================================================================

def load_params(file_path: str) -> dict:
    """
    Load parameters from params.yaml.
    """

    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug("Parameters loaded from %s", file_path)

        return params

    except FileNotFoundError:
        logger.error("Parameter file not found: %s", file_path)
        raise

    except yaml.YAMLError as e:
        logger.error("YAML Error: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected Error: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load training data.
    """

    try:
        df = pd.read_csv(file_path)

        logger.debug(
            "Data loaded from %s with shape %s",
            file_path,
            df.shape,
        )

        return df

    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise

    except Exception as e:
        logger.error("Error while loading data: %s", e)
        raise


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    params: dict
) -> RandomForestClassifier:
    """
    Train Random Forest model.
    """

    try:

        if len(x_train) != len(y_train):
            raise ValueError(
                "x_train and y_train must contain the same number of samples."
            )

        logger.debug(
            "Initializing RandomForestClassifier..."
        )

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            random_state=params["random_state"]
        )

        logger.debug(
            "Training model on %d samples...",
            len(x_train)
        )

        model.fit(x_train, y_train)

        logger.debug("Model training completed successfully.")

        return model

    except KeyError as e:
        logger.error("Missing parameter in params.yaml: %s", e)
        raise

    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise


def save_model(model, file_path: str):
    """
    Save trained model.
    """

    try:

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logger.debug(
            "Model saved successfully at %s",
            file_path
        )

    except Exception as e:
        logger.error("Error while saving model: %s", e)
        raise


# =============================================================================
# Main Function
# =============================================================================

def main():

    try:

        # Load parameters
        params = load_params(
            os.path.join(BASE_DIR, "params.yaml")
        )

        model_params = params["model_building"]

        # Load processed training data
        train_data = load_data(
            os.path.join(
                BASE_DIR,
                "data",
                "processed",
                "train_tfidf.csv",
            )
        )

        # Features and Target
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train model
        model = train_model(
            x_train=x_train,
            y_train=y_train,
            params=model_params,
        )

        # Save model
        save_model(
            model,
            os.path.join(
                BASE_DIR,
                "data",
                "models",
                "model.pkl",
            ),
        )

        logger.debug("Model Building Stage Completed Successfully.")

    except Exception as e:

        logger.error(
            "Failed to complete model building stage: %s",
            e,
        )

        raise


if __name__ == "__main__":
    main()