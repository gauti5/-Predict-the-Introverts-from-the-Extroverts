import sys
import os

from src.logging import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, Data_Transformation_Config
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

if __name__=='__main__':
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr, test_arr=data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))