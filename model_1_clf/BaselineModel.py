import random
import warnings
import datetime
import matplotlib
matplotlib.use("Agg")

import numpy as np
import utils.Helper as utils
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from sklearn import metrics
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from timeit import default_timer as timer
from sklearn.ensemble import IsolationForest

random.seed(111)


def log(text, end='\n'):
    utils.log(text, 'model_1.txt', end=end)


class BaselineModel:
    def __init__(self, v2w_min_cnt=1, v2w_window=10, v2w_size=45, block_size=10, block_step=1):
        self.__v2w_min_cnt = v2w_min_cnt
        self.__v2w_window = v2w_window
        self.__v2w_size = v2w_size
        self.__block_size = block_size
        self.__block_step = block_step
        self.__event_dictionary = None
        self.__clf = None
        self.__nu = 0.1
        self.__kernel = "rbf"
        self.__gamma = 0.1
        self.__n_estimators = 100
        self.__max_samples = 256
        self.__contamination = 0.1
        self.__threshold = None

    def set_svm_parameters(self, nu, kernel, gamma):
        self.__nu = nu
        self.__kernel = kernel
        self.__gamma = gamma

    def set_iforest_parameters(self, n_estimators, max_samples, contamination):
        self.__n_estimators = n_estimators
        self.__max_samples = max_samples
        self.__contamination = contamination

    def train(self, train_processes, mode='if', iter=0, dump=True):
        start = timer()
        log('===== TRAINING =====')
        log('===== Events processing =====')
        self.__events_processing(train_processes)
        log('===== Blocks processing =====')
        blocks, _ = self.__blocks_processing(train_processes)
        # visualisation of data
        # blocks_embedded = TSNE(n_components=2).fit_transform(blocks)
        # print('TNSE Shape:', blocks_embedded.shape, end='. ')
        # fig = plt.figure()
        # plt.scatter(blocks_embedded[:, 0], blocks_embedded[:, 1])
        # fig.savefig('pics/test.png')
        # print('Figure is saved in \'pics/test.png\'.')
        # print('Execution time to current point:', str(datetime.timedelta(seconds=timer() - start)))
        if mode == 'if':
            log('===== Isolation Forest fitting =====')
            clf = IsolationForest(n_estimators=self.__n_estimators, max_samples=self.__max_samples,
                                  contamination=self.__contamination, n_jobs=-1)
        elif mode == 'svm':
            log('===== OneClassSVM fitting =====')
            clf = OneClassSVM(nu=self.__nu, kernel=self.__kernel, gamma=self.__gamma)
        else:
            raise RuntimeError('Unknown mode: %s' % mode)
        clf.fit(blocks)
        self.__clf = clf
        log('Training done.')
        if dump:
            self.__dump_model(iter)
        log('Training is finished. Total time: %s' % str(datetime.timedelta(seconds=timer() - start)))

    def validation(self, normal, anomalous, iter=0, dump=True):
        log('===== VALIDATION =====')
        answers = [0] * len(normal) + [1] * len(anomalous)
        blocks, blocks_ends = self.__blocks_processing(normal + anomalous)
        blocks_predictions = list(self.__clf.predict(blocks))
        scores = []
        start = 0
        for end in blocks_ends:
            process = blocks_predictions[start:end]
            outlier_score = process.count(-1) / len(process)
            scores.append(outlier_score)
            start = end
        fpr, tpr, thresholds = metrics.roc_curve(answers, scores, pos_label=1, drop_intermediate=False)
        optimal_idx = np.argmax(tpr - fpr)
        self.__threshold = min(thresholds[optimal_idx], 1.)
        log('Selected threshold: %f' % self.__threshold)
        if dump:
            self.__dump_model(iter)

    def __dump_model(self, iter=0):
        filename = 'dumps/model_0' + ('' if iter == 0 else '_' + str(iter)) + '.pkl'
        joblib.dump((self.__event_dictionary, self.__block_size, self.__block_step, self.__clf, self.__threshold),
                    filename)
        log('Model is dumped into ' + filename + ' file.')

    def load_model(self, file):
        log('Model is loading from file %s...' % file)
        self.__event_dictionary, self.__block_size, self.__block_step, self.__clf, self.__threshold = joblib.load(file)

    def test(self, test_processes, block_step=1):
        log('===== TESTING =====')
        assert self.__threshold is not None, 'First pass through validation phase'
        self.__block_step = block_step
        blocks, blocks_ends = self.__blocks_processing(test_processes)
        blocks_predictions = list(self.__clf.predict(blocks))
        scores = []
        predictions = []
        start = 0
        for end in blocks_ends:
            process = blocks_predictions[start:end]
            # log('New process. Count of neg patterns: %d/%d (%f%%)' % (process.count(-1), len(process),
            #                                                           process.count(-1) / len(process)))
            outlier_score = process.count(-1) / len(process)
            scores.append(outlier_score)
            predictions.append(1 if outlier_score >= self.__threshold else 0)
            start = end
        return scores, predictions

    def __events_processing(self, processes):
        log('Training Word2Vec model...')
        model = Word2Vec(processes, sg=0, min_count=self.__v2w_min_cnt, window=self.__v2w_window, size=self.__v2w_size,
                         sample=1e-5)
        vocab_vectors = model[model.wv.vocab]
        self.__event_dictionary = {}
        for word, vec in zip(list(model.wv.vocab), vocab_vectors):
            self.__event_dictionary[word] = vec
        log('Done. Total dictionary length: %d' % len(list(model.wv.vocab)))

    def __blocks_processing(self, processes):
        assert self.__event_dictionary is not None
        log('Collecting and processing blocks...')
        blocks = []
        ends = []
        for process in processes:
            cnt = 0
            events_count = len(process)
            for start_pos in range(0, events_count, self.__block_step):
                end_pos = min(start_pos + self.__block_size, events_count)
                if end_pos - start_pos < self.__block_size:
                    continue
                block = process[start_pos:end_pos]
                block_sum = np.array([0.] * self.__v2w_size)
                for word in block:
                    block_sum += self.__event_dictionary.get(word, np.array([0.] * self.__v2w_size))
                blocks.append(block_sum)
                cnt += 1
            if cnt == 0:
                raise RuntimeError('Too small process: ' + str(process))
            ends.append(len(blocks))
        log('Done. Count of blocks: %d; count of processes: %d' % (len(blocks), len(ends)))
        return np.array(blocks), ends
