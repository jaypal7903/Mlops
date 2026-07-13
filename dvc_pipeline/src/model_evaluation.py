import numpy as np
import pandas as pd
import os
import logging
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Ensure log directory exists
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok= True)

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# create consol handler object
consol_handler = logging.StreamHandler()
consol_handler.setLevel('DEBUG')

# create file handler object
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# set message and set  in consol handler and file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consol_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(consol_handler)
    logger.addHandler(file_handler)

def load_model(file_path : str) :
    """
    Load the trained model from a file.
    """
    try :
        with open(file_path, 'rb') as file :
            model = pickle.load(file)
        logger.debug('the model is loaded form %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('The file not found: %s', file_path)
        raise
    except Exception as e :
        logger.error('Unexpected error occurred while loading the model: %s',e)
        raise

def load_data(file_path : str) -> pd.DataFrame :
    """
    Load the data from a CSV file.
    """
    try :
        df = pd.read_csv(file_path)
        logger.debug('Data is loeded from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e :
        logger.error('unexpected error occurred while loading the data: %s',e)
        raise

def model_evaluation(clf, x_test : np.ndarray , y_test : np.ndarray) -> dict :
    """
    Evaluate the model and return evaluation matrics.
    """
    try :
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        matrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
        }
        logger.debug('Model Evaluation matrix calculated.')
        return matrics_dict
    except Exception as e :
        logger.error('Unexpected error occurres during model evaluation')
        raise

def save_matrics(matrics : dict , file_path : str) -> None :
    """
    Save the evaluation matrics to a json file.
    """
    try :
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok= True)

        with open(file_path, 'w') as file :
            json.dump(matrics,file ,indent= 4)
        logger.debug('Matrix saved  to %s',file_path)
    except Exception as e :
        logger.error('Errors occurred while saving the metrics: %s',e)
        raise

def main() :
    try :
        clf = load_model("D:/Mlops/dvc_pipeline/data/models/model.pkl")
        test_data = load_data('D:/Mlops/dvc_pipeline/data/processed/test_tfidf.csv')

        x_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        matrics = model_evaluation(clf=clf, x_test= x_test, y_test= y_test)

        file_path = 'D:/Mlops/dvc_pipeline/data/reports/matrics.json'
        save_matrics(matrics= matrics,file_path= file_path)

    except Exception as e :
        logger.error('Unexpected error occurred during the model evaluation process: %s', e)
        print(f"Error : {e}")
        raise

if __name__ == '__main__' :
    main()