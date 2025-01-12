
from sensor.utils.main_utils import load_numpy_array_data
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
import os,sys
from xgboost import XGBClassifier
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object,load_object
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,precision_recall_curve
from numpy import mean
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV

class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise SensorException(e,sys)

    def train_model(self,x_train,y_train):
        try:
            max_depth = [5,10,15, 20, 25]
            min_child_weight= [1, 5, 10]
            n_estimators = [10,30,50,80,100,250]
            colsample_bytree = [0.3,0.5,0.7,1]
            subsample = [0.5,0.5,0.7,1]
        
            param = {'max_depth':max_depth,'min_child_weight':min_child_weight,'n_estimators':n_estimators, 'colsample_bytree':colsample_bytree,'subsample':subsample}
            clf = XGBClassifier(n_jobs=-1, random_state=42, scale_pos_weight = 1.7 )
            tuning = RandomizedSearchCV(estimator=clf,param_distributions=param,cv=3,scoring='f1_macro',n_jobs=-1,return_train_score=True,verbose=10)

           
            tuning.fit(x_train,y_train)
            best = tuning.best_params_
            print(best)

            # Best model with tuned hyperparameters
            best_XGB_model = tuning.best_estimator_
            calib_XGB = CalibratedClassifierCV(base_estimator=best_XGB_model, cv=5, method='sigmoid')
            calib_XGB.fit(x_train,y_train)
            return calib_XGB
        except Exception as e:
            raise SensorException(e,sys)
        
        
   

    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            

            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            classification_train_metric =  get_classification_score(y_true=y_train, y_pred=y_train_pred)

          

            if classification_train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")
            
           
            if classification_train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")
            
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

                     
            y_test_pred_prob= model.predict_proba(x_test)[:,1]

            precision, recall, thresholds = precision_recall_curve(y_test,y_test_pred_prob)
            def best_prob(precision, recall, thresholds):
                 f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
                 best_index = f1_scores.argmax()
                 return thresholds[best_index]
            best_threshold = best_prob(precision[:-1], recall[:-1], thresholds)
            print(f"Best probability threshold: {best_threshold}")



            #Overfitting and Underfitting
            diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)
            print(diff)
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")
            
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            sensor_model = SensorModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)

            #model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)