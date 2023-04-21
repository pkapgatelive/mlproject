import sys
#adding src to the system path
sys.path.insert(0, '/home/USERNAME/PATH/TO/src')
import os

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTranerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTranier:
    
    def __init__(self):
    
        self.model_trainer_config = ModelTranerConfig()
    
    def initiate_model_trainer(self, train_array,test_array,preprocessor_path):
        try:
            logging.info("Spliting traning and Test input Data")

        except Exception as e:
            raise CustomException(e,sys)
