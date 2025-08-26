import os 
import sys
import pandas 

from src.logging import logging
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTramsformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass

@dataclass 

class DataTransformationConfig:
    preprocessor_file_path=os.path.join("Artifacts", "Preprocessor.pkl")
    
class DataTransfornation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transfornation():
        try:
            logging.info("Data Transformation Started!!!")
            
            Num_cols=['id', 'Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size','Post_frequency']
            Cat_cols=['Stage_fear', 'Drained_after_socializing']
            
            Stage_fear_categories=['Yes', 'No']
            Drained_after_socializing_categories=['Yes', 'No']
            
            logging.info('Pipeline Started!!!')
            
            Num_Pipeline=Pipeline(
                steps=[
                    ('inputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            Cat_Pipeline=Pipeline(
                steps=[
                    ('inputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(categories=[Stage_fear_categories, Drained_after_socializing_categories], handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor=ColumnTramsformer([
                ('num pipeline', Num_Pipeline, Num_cols),
                ('cate pipeline', Cat_Pipeline, Cat_cols)
            ])
            
            return preprocessor
        except Exception as e:
            logging.info("Exception occured during the data transformation!!!")
            raise CustomException(e,sys)

