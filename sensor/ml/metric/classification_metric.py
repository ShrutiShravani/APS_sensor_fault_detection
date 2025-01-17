from sensor.entity.artifact_entity import ClassificationMetricArtifact
from sensor.exception import SensorException
import os,sys
from sklearn.metrics import f1_score,precision_score,recall_score


def get_classification_score(y_true,y_pred) ->ClassificationMetricArtifact:
    try:
        model_f1_score= f1_score(y_true,y_pred, average = 'macro')
        model_precision_score= precision_score(y_true,y_pred)
        model_recall_score =recall_score(y_true,y_pred)

        classfication_metric=ClassificationMetricArtifact(f1_score=model_f1_score,precision_score=model_precision_score,recall_score=model_recall_score)
        print(model_f1_score)
        print(model_precision_score)
        print(model_recall_score)
        
        return classfication_metric
    except Exception as e:
        raise SensorException(e,sys)




