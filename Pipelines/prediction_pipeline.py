import os
import sys

import pandas as pf

from src.logging import logging
from src.exception import CustomException
from src.utils import load_object

from pathlib import Path

class predict_pipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            preprocessor_path=os.path.join("Artifacts", "preprocessor.pkl")
            model_path=os.path.join("Artifacts", "model.pkl")
            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            data_scaled=preprocessor.transform(features)
            
            pred=model.predict(data_scaled)
            
            return pred
        
        except Exception as e:
            logging.info("Error occured during the prediction pipeline")
            raise CustomException(e,sys)