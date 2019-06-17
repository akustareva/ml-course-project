import random

from model_1_clf.BaselineModel import BaselineModel, log
from model_1_clf.SearchForOptimalParameters import testing
from utils.Helper import train_validation_test_split

random.seed(111)

if __name__ == '__main__':
    _, _, normal_test, _, anomalous_test = train_validation_test_split(['clean'], ['infected'], '_model_1_text', log)
    model = BaselineModel()
    model.load_model('dumps/model_1_33.pkl')
    testing(model, normal_test, anomalous_test, 12)
