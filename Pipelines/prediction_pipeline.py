import os
import sys

import pandas as pd

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
        
    class CustomData:
        def __init__(self,
                     id:int,
                     Time_spent_Alone:float,
                     Stage_fear:object,
                     Social_event_attendance:float,
                     Going_outside:float,
                     Drained_after_socializing:object,
                     Friends_circle_size:float,
                     Post_frequency:float):
            self.id=id
            self.Time_spent_Alone=Time_spent_Alone
            self.Stage_fear=Stage_fear
            self.Social_event_attendance=Social_event_attendance
            self.Going_outside=Going_outside
            self.Drained_after_socializing=Drained_after_socializing
            self.Friends_circle_size=Friends_circle_size
            self.Post_frequency=Post_frequency
            
        def get_data_as_dataframe(self):
            try:
                custom_data_input_dict={
                    'id':[self.id],
                    'Time_spent_Alone':[self.Time_spent_Alone],
                    'Stage_fear':[self.Stage_fear],
                    'Social_event_attendance':[self.Social_event_attendance],
                    'Going_outside':[self.Going_outside],
                    'Drained_after_socializing':[self.Drained_after_socializing],
                    'Drained_after_socializing':[self.Drained_after_socializing],
                    'Friends_circle_size':[self.Friends_circle_size],
                    'Post_frequency':[self.Post_frequency]
                }
                
                df=pd.DataFrame(custom_data_input_dict)
                return df
            except Exception as e:
                logging.info("Error ocuured during the prediction pipeline!!!")
                raise CustomException(e,sys)