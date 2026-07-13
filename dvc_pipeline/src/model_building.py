import numpy as np
import pandas as pd
import os
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier


# Ensure log directory exists
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok= True)

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

# create the consol handler object
consol_handler = logging.StreamHandler()
consol_handler.setLevel('DEBUG')

# create the object of file handler 
log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# set the message formmet and set in consol handler and file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consol_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(consol_handler)
logger.addHandler(file_handler)

def load_data(file_path : str) -> pd.DataFrame :
    """
    Load the data from a CSV  file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data Loaded from the %s with the shape %s',file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file : %s', e)
        raise
    except FileNotFoundError as e :
        logger.error('File Not found : %s', e)
        raise
    except Exception as e :
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(x_train : np.ndarray , y_train : np.ndarray) -> RandomForestClassifier :
    """
    Train the RandomForest Model.

    x_train : training features
    y_train : training labels
    return : RandomforestClassifire
    """
    try :
        if x_train.shape[0] != y_train.shape[0] :
            raise ValueError('The number of sample in x_train and y_train is must be same.')
        
        logger.debug('Intialized the RandomForest model')
        clf = RandomForestClassifier(n_estimators=22, random_state=2)

        logger.debug('Model training started with %d samples', x_train.shape[0])
        clf.fit(x_train, y_train)
        logger.debug('Model training completed.')

        return clf
    
    except ValueError as e :
        logger.error('Value error during the training : %s', e)
        raise
    except Exception as e :
        logger.error('Error during model training %s', e)
        raise

def save_model(model: RandomForestClassifier, file_path : str) -> None :
    """
    Saved the trained model to file.
    """
    try :
        # Ensure the directory is exists
        os.makedirs(os.path.dirname(file_path), exist_ok= True)

        with open(file_path, 'wb') as file :
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)

    except FileNotFoundError as e :
        logger.error('File not found :%s', e)
        raise
    except Exception as e :
        logger.error('Error occurred while saving the model: %s',e)
        raise

def main() :
    try :
        train_data = load_data("D:/Mlops/dvc_pipeline/data/processed/train_tfidf.csv")
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(x_train=x_train, y_train= y_train)

        model_save_path = 'D:/Mlops/dvc_pipeline/data/models/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e :
        logger.error('Failed to complete the model buiding process : %s', e)
        print(f"Error : {e}")

if __name__ == '__main__' :
    main()