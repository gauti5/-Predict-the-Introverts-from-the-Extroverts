import os 
import sys
import pandas as pd
import numpy as np

from src.logging import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass

@dataclass 

class Data_Transformation_Config:
    preprocessor_file_path=os.path.join('Artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=Data_Transformation_Config()
        
    def get_data_transfornation(self):
        try:
            logging.info("Data Transformation Started!!!")
            
            Num_cols=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size','Post_frequency']
            Cat_cols=['Stage_fear', 'Drained_after_socializing']
            
            Stage_fear_categories=['Yes', 'No']
            Drained_after_socializing_categories=['Yes', 'No']
            
            logging.info('Pipeline Started!!!')
            
            Num_Pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            Cat_Pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(categories=[Stage_fear_categories, Drained_after_socializing_categories], handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor=ColumnTransformer([
                ('num pipeline', Num_Pipeline, Num_cols),
                ('cate pipeline', Cat_Pipeline, Cat_cols)
            ])
            
            return preprocessor
        except Exception as e:
            logging.info("Exception occured during the data transformation!!!")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            preprocessor_obj=self.get_data_transfornation()
            
            logging.info("read the training data and testing data")
            
            logging.info(f"Training DataFrame : \n{train_df.head(5).to_string()}")
            logging.info(f"Testing DataFrame : \n{test_df.head(5).to_string()}")
            
            input_features_train_df=train_df.drop('Personality', axis=1)
            target_features_train_df=train_df['Personality']
            
            input_features_test_df=test_df.drop('Personality', axis=1)
            target_features_test_df=test_df['Personality']
            
            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessor_obj.transform(input_features_test_df)
            
            
            # Combines features (X) and target (y) into one array for both train and test.
            # np.c_[] concatenates them column-wise
            train_arr=np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr=np.c_[input_features_test_arr, np.array(target_features_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor_obj
            )
            logging.info("preprocessor pickle file saved!!")
            
            return(
                train_arr, test_arr
            )
            
        except Exception as e:
            logging.info("Error occured during data transformation!!")
            raise CustomException(e, sys)
        
    
"""
Raw CSV → Pandas DataFrame

Split into Features (X) + Target (y)

Pass X through Preprocessor (Num + Cat pipelines)

Get transformed features

Add target column back

Return final numpy arrays for Train & Test

Save preprocessor for later use
"""
