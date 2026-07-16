import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# Ensure the "logs" directory exists
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# create the object of consol handler
consol_handler = logging.StreamHandler()
consol_handler.setLevel('DEBUG')

# create the object of the file handler
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# set the message format and set in consol_handler and file_handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consol_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add file_handler and consol_handler in logger object
logger.addHandler(consol_handler)
logger.addHandler(file_handler)

def load_params(file_path : str) -> dict :
    """
    Load the parameter form yaml file.
    """
    try :
        with open(file_path , 'r') as file :
            params = yaml.safe_load(file)
        logger.debug('params retrived form %s', file_path)
        return params
    except FileNotFoundError as e :
        logger.error('File not found : %s', file_path)
        raise
    except yaml.YAMLError as e :
        logger.error('YAML error: %s', e)
        raise
    except Exception as e :
        logger.error('unexpected error: %s', e)
        raise

# The data logging function
def load_data(data_url : str) -> pd.DataFrame :
    "load data from csv file"
    try :
        df = pd.read_csv(data_url)
        logger.debug('data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e :
        logger.error('Filed to parse the csv file : %s',e)
        raise
    except Exception as e :
        logger.error('Unexpected error occurs while logging the data : %s', e)
        raise

# The data preprocessing function
def preprocess_data(df : pd.DataFrame) -> pd.DataFrame :
    "preprocess the data"
    try :
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace= True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'},inplace=True)
        logger.debug('data preprocessing completed')
        return df
    except KeyError as e :
        logger.error('Missing columns in the data frame: %s', e)
    except Exception as e :
        logger.error('Unexpected eror during preprocessing : %s', e)

# The save data function
def save_data(train_data : pd.DataFrame, test_data : pd.DataFrame, data_path : str) -> None :
    "save the train and test datasets"
    try :
       # Create the data/raw directory
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index= False)
        logger.debug('Train and Test data saved to %s', raw_data_path)
    except Exception as e :
        logger.error('Unexpected error occurred while saving the data : %s', e)

# The main function
def main() :
    try :
        # test_size = 0.2
        params = load_params('D:\Mlops\dvc_pipeline\params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = os.path.join(BASE_DIR, 'experiments', 'spam.csv')
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data , test_data = train_test_split(final_df, test_size= test_size, random_state=2)
        save_data(train_data= train_data, test_data= test_data, data_path=os.path.join(BASE_DIR, "data"))
    except Exception as e :
        logger.error('Filed to completed the data ingestion process: %s', e)
        print(f"Error: {e}")

# calling the main function
if __name__ == '__main__' :
    main()