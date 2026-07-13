import pandas as pd
import logging
import os
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the log directory exists
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(BASE_DIR , 'logs')
os.makedirs(log_dir, exist_ok= True)

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# consol handler object
consol_handler = logging.StreamHandler()
consol_handler.setLevel('DEBUG')

# file handler object
log_file_path = os.path.join(log_dir , 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# set the message format and set in consol handler and file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consol_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(consol_handler)
logger.addHandler(file_handler)

def transform_text(text) :
    """
    Transform the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation , and stemming.
    """
    ps = PorterStemmer()
    # convert to lowercase
    text = text.lower()
    # tokenize the word
    text = word_tokenize(text)
    # Remove non - alphanumerical tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # stem the words
    text = [ps.stem(word) for word in text]
    # join the tokens back into single string
    return " ".join(text)

def preprocess_df(df, text_column = 'text', target_column = 'target') :
    """
    Preprocess the DataFrame by encoding the target column , removing duplicates, and transforming the text column. 
    """
    try :
        logger.debug('starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('target column encoded')

        # Remove Duplicate rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicates removed')

        # Applied to transformation for specified column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e :
        logger.error('Column not found: %s', e)
        raise
    except Exception as e :
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column = 'text', target_column = 'target'):
    """
    Main function is load the raw data , preprocess it and save the processed data.
    """
    try :
        # Featch the raw data
        train_data = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "train.csv"))
        test_data = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "test.csv"))
        logger.debug('Data loaded Properly')
        
        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column= text_column, target_column= target_column)
        test_processed_data = preprocess_df(test_data, text_column= text_column, target_column= target_column)

        # Store the data inside data/processed
        data_path = os.path.join(BASE_DIR, "data", "interim")
        os.makedirs(data_path, exist_ok= True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed_data.csv"), index= False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed_data.csv"), index= False)

        logger.debug('Processed data saved to %s', data_path)

    except FileNotFoundError as e :
        logger.error('File not found : %s ' ,e)
    except pd.errors.EmptyDataError as e :
        logger.error('No data: %s ', e)
    except Exception as e :
        logger.error('Failed to complete the data transformation process: %s ', e)
        print(f"Error : {e}")

if __name__ == '__main__':
    main()