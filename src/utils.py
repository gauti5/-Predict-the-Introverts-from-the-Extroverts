import os
import sys
from pathlib import Path
import pickle

from src.logging import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.info("Error occured while saving the object")
        raise CustomException(e,sys)