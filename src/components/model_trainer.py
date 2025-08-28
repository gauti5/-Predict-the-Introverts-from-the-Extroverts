import os
import sys

import pandas as pd
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from src.logging import logging
from src.exception import CustomException

from src.utils import save_object

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


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
                'AdaBoost Classifier': AdaBoostClassifier(),
                'Gradinet Boostoing Classifier': GradientBoostingClassifier(),
                'K Neigbors Classifier': KNeighborsClassifier()
            }