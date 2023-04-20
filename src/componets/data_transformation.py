import os
import sys
# adding src to the system path
sys.path.insert(0, '/home/USERNAME/PATH/TO/src')

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomeException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps= [
                ("imputer",SimpleImputer(strategy="medium")),
                ("scaler", StandardScaler())
                ])
            
            cat_pipeline = Pipeline(

                steps= [
                ("imputer", SimpleImputer(strategy="most_frequent"))
                ("one_hot_encoder", OneHotEncoder()),
                ('scaler', StandardScaler())
                ])
            
            logging.info("numerical columns standad scaling completed") 
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ])

            return preprocessor

        except Exception as e:
            raise CustomeException(e,sys)
