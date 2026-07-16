import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import yaml

# Ensure log directory exists
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok= True)

# Logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

# create consol handler object
consol_handler = logging.StreamHandler()
consol_handler.setLevel('DEBUG')

# create file handler object
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# set message formatt and set in consol handler and file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consol_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(consol_handler)
logger.addHandler(file_handler)

def load_params(file_path : str) -> dict :
    """
    Load the parameter form YAML file.
    """
    try :
        with open(file_path , 'r') as file :
            params = yaml.safe_load(file)
        logger.debug('params retrived from %s', file_path)
        return params
    except FileNotFoundError as e :
        logger.error('File not found: %s',file_path)
        raise
    except yaml.YAMLError as e :
        logger.error('yaml error : %s', e)
        raise
    except Exception as e :
        logger.error('unexpected error :%s', e)
        raise

def load_data(file_path : str) -> pd.DataFrame :
    """Load data from a CSV file ."""
    try :
        df = pd.read_csv(file_path)
        df.fillna('', inplace=  True)
        logger.debug('Data loaded and NaNs filed from %s', file_path)
        return df
    except pd.errors.ParserError as e :
        logger.error('Failed to parse the CSV file: %s ', e)
        raise
    except Exception as e :
        logger.error('Unexpected error occurs while loading the data: %s ', e)
        raise

def apply_tfidf(train_data : pd.DataFrame , test_data : pd.DataFrame, max_features : int) -> tuple :
    """apply Tfidf to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features= max_features)

        x_train = train_data['text'].astype(str).values
        y_train = train_data['target'].values

        x_test = test_data['text'].astype(str).values
        y_test = test_data['target'].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray(), columns= vectorizer.get_feature_names_out())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray(), columns= vectorizer.get_feature_names_out())
        test_df['label'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df , test_df
    except Exception as e:
        logger.error('error during tfidf transformation: %s' , e)
        raise

def save_data(df: pd.DataFrame, file_path : str) -> None :
    """Save the data frame to CSV file."""
    try :
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        df.to_csv(file_path, index= False)
        logger.debug('Data saved to %s', file_path) 

    except Exception as e :
        logger.error('unexpected error occurs while saving the data: %s ', e)

def main() :
    try:
        # max_feature = 50
        params = load_params('D:\Mlops\dvc_pipeline\params.yaml')
        max_feature = params['feature_engineering']['max_feature']

        train_data = load_data("D:/Mlops/dvc_pipeline/data/interim/train_processed_data.csv")
        test_data = load_data("D:/Mlops/dvc_pipeline/data/interim/test_processed_data.csv")

        train_df , test_df = apply_tfidf(train_data=train_data, test_data=test_data,max_features=max_feature)

        save_data(train_df, os.path.join(BASE_DIR , 'data', 'processed', 'train_tfidf.csv'))
        save_data(test_df, os.path.join(BASE_DIR, 'data', 'processed', 'test_tfidf.csv'))

    except Exception as e :
        logger.error('Failed to completed the feature engineering process: %s ',e)
        print(f"Error : {e}")

if __name__ == '__main__' :
    main()