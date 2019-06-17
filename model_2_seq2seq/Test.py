import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
import keras.backend as K

from sklearn import metrics
from keras.models import load_model
from sklearn.externals import joblib
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from utils.GetProcesses import *
from model_2_seq2seq.Train import BLOCK_LEN, EVENTS_CNT, get_data, create_model, log


def events_to_one_hot(processes):
    return to_categorical(processes, EVENTS_CNT)


def test_symbol_acc(y_true, y_pred):
    return np.mean(y_true == y_pred, axis=1)


def get_detailed_procs_data(directories):
    processes = get_splitted_processes(directories, '_model_2_text')
    # processes = processes[:45]
    blocks = []
    ends = []
    for process in processes:
        cnt = 0
        events_count = len(process)
        for start_pos in range(0, events_count, BLOCK_LEN):
            end_pos = min(start_pos + BLOCK_LEN, events_count)
            if end_pos - start_pos < BLOCK_LEN:
                continue
            block = list(map(int, process[start_pos:end_pos]))
            blocks.append(block)
            cnt += 1
        if cnt > 0:
            ends.append(len(blocks))
    return np.array(blocks), ends


def get_procs_outlier_score(blocks_scores, ends, block_threshold):
    scores = []
    start = 0
    for end in ends:
        if start == end:
            continue
        process = blocks_scores[start:end]
        neg_cnt = 0
        for block_score in process:
            if block_score < block_threshold:
                neg_cnt += 1
        scores.append(neg_cnt * 1. / len(process))
        start = end
    return scores


def get_block_threshold(clean_dirs, infected_dirs):
    print('===== BLOCK THRESHOLD =====')
    clean_processes = get_data(clean_dirs)
    print('All clean data shape: %s' % str(clean_processes.shape))
    clean_validation, clean_test = train_test_split(clean_processes, train_size=0.029, test_size=0.015, shuffle=True)
    print('Validation shape: %s' % str(clean_validation.shape))
    print('Test shape: %s' % str(clean_test.shape))
    infected_processes = get_data(infected_dirs)
    print('All infected data shape: %s' % str(infected_processes.shape))
    infected_validation, infected_test = train_test_split(infected_processes, train_size=0.95, test_size=0.05,
                                                          shuffle=True)
    print('Validation shape: %s' % str(infected_validation.shape))
    print('Test shape: %s' % str(infected_test.shape))

    clean_pred_probabilities = model.predict(events_to_one_hot(clean_validation), batch_size=32, verbose=1, steps=None)
    clean_predictions = np.argmax(clean_pred_probabilities, axis=-1)
    print('Clean predictions shape: %s' % str(clean_predictions.shape))
    clean_blocks_scores = test_symbol_acc(clean_validation, clean_predictions)
    print('Clean blocks scores shape: %s' % str(clean_blocks_scores.shape))
    print('Clean blocks scores mean: %f' % clean_blocks_scores.mean())
    infected_pred_probabilities = model.predict(events_to_one_hot(infected_validation), batch_size=32, verbose=1,
                                                steps=None)
    infected_predictions = np.argmax(infected_pred_probabilities, axis=-1)
    print('Infected predictions shape: %s' % str(infected_predictions.shape))
    infected_blocks_scores = test_symbol_acc(infected_validation, infected_predictions)
    print('Infected blocks scores shape: %s' % str(infected_blocks_scores.shape))
    print('Infected blocks scores mean: %f' % infected_blocks_scores.mean())

    answers = [1] * len(clean_blocks_scores) + [0] * len(infected_blocks_scores)
    scores = list(clean_blocks_scores) + list(infected_blocks_scores)
    print('Answers len: %d; scores len: %d' % (len(answers), len(scores)))
    print('ROC AUC score: %f' % roc_auc_score(answers, scores))
    fpr, tpr, thresholds = metrics.roc_curve(answers, scores, pos_label=1, drop_intermediate=False)
    optimal_idx = np.argmax(tpr - fpr)
    block_threshold = min(thresholds[optimal_idx], 1.)
    print('Block threshold: %f' % block_threshold)
    return block_threshold


def get_process_threshold(clean_dirs, infected_dirs, block_threshold):
    print('===== PROCESS THRESHOLD =====')
    clean_processes, clean_ends = get_detailed_procs_data(clean_dirs)
    print('Clean blocks cnt: %d, total processes cnt: %d' % (len(clean_processes), len(clean_ends)))
    infected_processes, infected_ends = get_detailed_procs_data(infected_dirs)
    print('Infected blocks cnt: %d, total processes cnt: %d' % (len(infected_processes), len(infected_ends)))
    clean_pred_probabilities = model.predict(events_to_one_hot(clean_processes), batch_size=32, verbose=1, steps=None)
    clean_predictions = np.argmax(clean_pred_probabilities, axis=-1)
    print('Clean predictions shape: %s' % str(clean_predictions.shape))
    clean_blocks_scores = test_symbol_acc(clean_processes, clean_predictions)
    print('Clean blocks scores shape: %s' % str(clean_blocks_scores.shape))
    infected_pred_probabilities = model.predict(events_to_one_hot(infected_processes), batch_size=32, verbose=1,
                                                steps=None)
    infected_predictions = np.argmax(infected_pred_probabilities, axis=-1)
    print('Infected predictions shape: %s' % str(infected_predictions.shape))
    infected_blocks_scores = test_symbol_acc(infected_processes, infected_predictions)
    print('Infected blocks scores shape: %s' % str(infected_blocks_scores.shape))
    clean_outlier_scores = get_procs_outlier_score(clean_blocks_scores, clean_ends, block_threshold)
    print('Count of clean processes: %d' % len(clean_outlier_scores))
    infected_outlier_scores = get_procs_outlier_score(infected_blocks_scores, infected_ends, block_threshold)
    print('Count of infected processes: %d' % len(infected_outlier_scores))
    proc_answers = [0] * len(clean_outlier_scores) + [1] * len(infected_outlier_scores)
    proc_scores = clean_outlier_scores + infected_outlier_scores
    assert len(proc_answers) == len(proc_scores)
    print('ROC AUC score: %f' % roc_auc_score(proc_answers, proc_scores))
    fpr, tpr, thresholds = metrics.roc_curve(proc_answers, proc_scores, pos_label=1, drop_intermediate=False)
    optimal_idx = np.argmax(tpr - fpr)
    proc_threshold = min(thresholds[optimal_idx], 1.)
    print('Process threshold: %f' % proc_threshold)
    return proc_threshold


def testing(clean_dirs, infected_dirs, block_threshold, process_threshold):
    log('===== TESTING =====')
    clean_processes, clean_ends = get_detailed_procs_data(clean_dirs)
    infected_processes, infected_ends = get_detailed_procs_data(infected_dirs)
    clean_pred_probabilities = model.predict(events_to_one_hot(clean_processes), batch_size=32, verbose=1, steps=None)
    clean_predictions = np.argmax(clean_pred_probabilities, axis=-1)
    clean_blocks_scores = test_symbol_acc(clean_processes, clean_predictions)
    infected_pred_probabilities = model.predict(events_to_one_hot(infected_processes), batch_size=32, verbose=1,
                                                steps=None)
    infected_predictions = np.argmax(infected_pred_probabilities, axis=-1)
    infected_blocks_scores = test_symbol_acc(infected_processes, infected_predictions)
    clean_outlier_scores = get_procs_outlier_score(clean_blocks_scores, clean_ends, block_threshold)
    infected_outlier_scores = get_procs_outlier_score(infected_blocks_scores, infected_ends, block_threshold)
    procs_answers = [0] * len(clean_outlier_scores) + [1] * len(infected_outlier_scores)
    procs_scores = clean_outlier_scores + infected_outlier_scores
    procs_preds = []
    for i, proc_score in enumerate(procs_scores):
        # log(str(i) + ' ' + str(proc_score), console=False)
        if proc_score >= process_threshold:
            procs_preds.append(1)
        else:
            procs_preds.append(0)
    log('Precision score: %f' % precision_score(procs_answers, procs_preds, pos_label=1, average='binary'))
    log('Recall score: %f' % recall_score(procs_answers, procs_preds, pos_label=1, average='binary'))
    log('F1-score: %f' % f1_score(procs_answers, procs_preds, pos_label=1, average='binary'))
    log('Report (average=\'weighted\'):')
    log(str(classification_report(procs_answers, procs_preds, target_names=['normal', 'anomalous'])))


if __name__ == '__main__':
    model = create_model()
    weights_filename = 'dumps/model_2_8_weights.h5'
    thresholds_filename = weights_filename.replace('weights', 'thresholds')
    thresholds_filename = thresholds_filename.replace('h5', 'pkl')
    log('Loading weights from %s...' % weights_filename)
    model.load_weights(weights_filename)

    # block_threshold = get_block_threshold(['clean'], ['infected'])  # val set 2
    # process_threshold = get_process_threshold(['clean'], ['infected'], block_threshold)  # val set 3
    # joblib.dump((block_threshold, process_threshold), thresholds_filename)

    log('Loading thresholds from %s...' % thresholds_filename)
    block_threshold, process_threshold = joblib.load(thresholds_filename)
    log(str(process_threshold))
    testing(['clean'], ['infected'], block_threshold, process_threshold)  # test set
