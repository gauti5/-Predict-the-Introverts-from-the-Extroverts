import os
import sys

import pandas as pd
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from src.logging import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("Artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        
        
        try:
            logging.info("Splitting the data into training and testing set!!!")
            
            X_train, y_train, X_test, y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models={
                'Logistic Regression': LogisticRegression(),
                'Bernoulli NB': BernoulliNB(),
                'Support Vector Classifier': SVC(),
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'Random Forest Classifier': RandomForestClassifier(),
                'AdaBoost Classifier': AdaBoostClassifier(algorithm="SAMME"),
                'Gradinet Boostoing Classifier': GradientBoostingClassifier(),
                'K Neigbors Classifier': KNeighborsClassifier()
            }
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            print(model_report)
            
            print("/n====================================================================")
            
            logging.info(f"Model Report : {model_report}")
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            print(f"Best Model Found !!, model name : {best_model}, Accuracy Score : {best_model_score}")
            
            print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            
            
            logging.info(f"Best Model Found, Model Name : {best_model}, Accuracy Score : {best_model_score}")
            
            if best_model_score<0.7:
                raise CustomException("Best Model Not Found!!!")
            logging.info("Best Model Found!!!")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.info("Error occured during the model training!!!")
            raise CustomException(e,sys)