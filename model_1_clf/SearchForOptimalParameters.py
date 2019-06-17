import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from utils.GetProcesses import *
from utils.Helper import train_validation_test_split
from model_1_clf.BaselineModel import BaselineModel, log

random.seed(111)


def testing(model, clean, infected, block_step):
    scores, predictions = model.test(clean + infected, block_step=block_step)
    answers = [0] * len(clean) + [1] * len(infected)
    log('===== Testing results =====')
    log('ROC AUC score: %f' % roc_auc_score(answers, scores))
    # log('Accuracy score: %f' % accuracy_score(answers, predictions))
    log('Precision score: %f' % precision_score(answers, predictions, pos_label=1, average='binary'))
    log('Recall score: %f' % recall_score(answers, predictions, pos_label=1, average='binary'))
    log('F1-score: %f' % f1_score(answers, predictions, pos_label=1, average='binary'))
    log('Report (average=\'weighted\'):')
    log(str(classification_report(answers, predictions, target_names=['normal', 'anomalous'])))


if __name__ == '__main__':
    iteration = 0
    clean_processes = get_splitted_processes(['clean'], '_model_0_text')
    infected_processes = get_splitted_processes(['infected'], '_model_0_text')
    normal_train, normal_validation, normal_test, anomalous_validation, anomalous_test = \
        train_validation_test_split(clean_processes, infected_processes, log)
    for block_size in [12, 8]:
        # for nu in [0.0001, 0.005, 0.1]:
        #     for kernel in ['rbf', 'poly']:
        #         for gamma in [0.00001, 0.001, 0.1, 3]:  # 0.01,
        #             iteration += 1
        #             log('========== NEW MODEL #%d ==========' % iteration)
        #             log('nu=%f; kernel=%s; gamma=%f' % (nu, kernel, gamma))
        #             model = BaselineModel(block_size=12, block_step=12)
        #             model.set_svm_parameters(nu=nu, kernel=kernel, gamma=gamma)
        #             model.train(normal_train, mode='svm', dump=False)
        #             model.validation(normal_validation, anomalous_validation, iter=iteration)
        #             testing(model, normal_test, anomalous_test, block_step=block_size)
        for n_estimators in [2000, 1000, 3000, 500]:
            for max_samples in [0.2, 0.3]:
                for contamination in [0.001, 0.01, 0.1, 0.0001]:
                    iteration += 1
                    log('========== NEW MODEL #%d ==========' % iteration)
                    log('block_size = %d; n_estimators=%d; max_samples=%f; contamination=%f' %
                        (block_size, n_estimators, max_samples, contamination))
                    model = BaselineModel(block_size=block_size, block_step=block_size)
                    model.set_iforest_parameters(n_estimators=n_estimators, max_samples=max_samples,
                                                 contamination=contamination)
                    model.train(normal_train, mode='if', dump=False)
                    model.validation(normal_validation, anomalous_validation, iter=iteration)
                    testing(model, normal_test, anomalous_test, block_step=block_size)
