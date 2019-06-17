import os
# import importlib
# os.environ['KERAS_BACKEND']='tensorflow'
# importlib.reload(backend)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import seq2seq
import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
import keras.backend as K
import utils.Helper as utils
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from seq2seq.models import Seq2Seq
from keras.utils import to_categorical
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Input, Dense, TimeDistributed, Activation
from keras.callbacks import LearningRateScheduler, TerminateOnNaN, LambdaCallback, ModelCheckpoint

from utils.GetProcesses import *
from dictionary.Dictionary import operations_count, results_count, paths_count

config = K.tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=5)
config.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=config))

BLOCK_LEN = 25
EVENTS_CNT = operations_count * results_count * paths_count  # 462


def string_as_array(event):
    return np.expand_dims(np.array(list(event), dtype=np.float32), axis=1)


def get_data(directories):
    blocks = []
    processes = get_splitted_processes(directories, '_model_2_text')
    for process in processes:
        start = 0
        while True:
            end = start + BLOCK_LEN
            if end > len(process):
                break
            block = list(map(int, process[start:end]))
            blocks.append(block)
            start = end
    print('Total processes cnt:', len(processes))
    return np.array(blocks)


def symbol_acc(y_true, y_pred):
    y_true = K.argmax(y_true)
    y_pred = K.argmax(y_pred)
    is_equal = K.cast(K.equal(y_true, y_pred), K.floatx())
    return K.mean(is_equal)


def seq_acc(y_true, y_pred):
    y_true = K.argmax(y_true)
    y_pred = K.argmax(y_pred)
    is_equal = K.equal(y_true, y_pred)
    return K.mean(K.all(is_equal, -1))


def get_leaning_rate(epoch):
    if epoch < 9:
        rate = 0.001
    elif epoch < 17:
        rate = 0.0005
    elif epoch < 33:
        rate = 0.0001
    elif epoch < 49:
        rate = 0.00005
    elif epoch < 81:
        rate = 0.00001
    else:
        rate = 0.000005
    return rate


def create_model():
    # input = Input(shape=(BLOCK_LEN,))
    # embedded = Embedding(input_dim=EVENTS_CNT + 1, input_length=BLOCK_LEN, output_dim=400)(input)
    # emb_model = Model(input, embedded)
    # print(emb_model.summary())

    seq_model = Seq2Seq(batch_input_shape=(None, BLOCK_LEN, EVENTS_CNT), hidden_dim=56, output_length=BLOCK_LEN,
                        output_dim=EVENTS_CNT, depth=1)  # , teacher_force=True, unroll=True
    # print(seq_model.summary())

    model = Sequential()
    # model.add(emb_model)
    model.add(seq_model)
    # model.add(Activation('softmax'))
    model.add(Dense(EVENTS_CNT, activation='softmax'))
    # model.summary()
    return model


def batch_generator(x_array, batch_size):
    while True:
        for i in range(len(x_array) // batch_size):
            x = x_array[i * batch_size:(i + 1) * batch_size]
            y = to_categorical(x, EVENTS_CNT)
            yield (y, y)


def log(text, end='\n', console=True, file=True):
    utils.log(text, 'model_2.txt', end=end, console=console, file=file)


if __name__ == '__main__':
    n_iter = 34
    clean_processes = get_data(['clean_part_1', 'clean_part_2'])
    log('All clean processes shape: %s' % str(clean_processes.shape))
    train, validation = train_test_split(clean_processes, test_size=0.1, shuffle=True)
    log('Train data shape: %s' % str(train.shape))
    log('Validation data shape: %s' % str(validation.shape))

    model = create_model()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc', symbol_acc, seq_acc])

    log_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: log('Epoch: %d. loss: %f - acc: %f - symbol_acc: %f - seq_acc: %f - '
                                             'val_loss: %f - val_symbol_acc: %f - val_seq_acc: %f' %
                                             (epoch + 1, logs['loss'], logs['acc'], logs['symbol_acc'], logs['seq_acc'],
                                              logs['val_loss'], logs['val_symbol_acc'], logs['val_seq_acc']),
                                             console=False))
    filename_for_weights = 'dumps/model_2' + ('' if n_iter == 0 else '_' + str(n_iter)) + '_weights.h5'
    checkpoint = ModelCheckpoint(filename_for_weights, save_best_only=True, save_weights_only=True)
    custom_lr_scheduler = LearningRateScheduler(get_leaning_rate)
    batch_size = 32
    train_steps = len(train) // batch_size
    val_steps = len(validation) // batch_size
    log('Steps per epoch: %d for train, %d for validation' % (train_steps, val_steps))
    history_callback = model.fit_generator(batch_generator(train, batch_size), steps_per_epoch=train_steps,
                                           validation_data=batch_generator(validation, batch_size),
                                           validation_steps=val_steps, epochs=90, verbose=1,
                                           callbacks=[log_callback, custom_lr_scheduler, TerminateOnNaN(), checkpoint])
    filename = 'dumps/model_2' + ('' if n_iter == 0 else '_' + str(n_iter)) + '.h5'
    model.save(filename)
    log('Model is dumped into \'%s\' file.' % filename)
    model.save_weights(filename_for_weights)
    log('Model weights are dumped into \'%s\' file.' % filename_for_weights)

    loss = np.array(history_callback.history['loss'])
    smbl_acc = np.array(history_callback.history['symbol_acc'])
    sq_acc = np.array(history_callback.history['seq_acc'])
    val_loss = np.array(history_callback.history['val_loss'])
    val_smbl_acc = np.array(history_callback.history['val_symbol_acc'])
    val_sq_acc = np.array(history_callback.history['val_seq_acc'])

    fig = plt.figure()
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    loss_filename = 'pics/loss' + ('' if n_iter == 0 else '_' + str(n_iter)) + '.png'
    fig.savefig(loss_filename)
    plt.close(fig)
    log('Loss visualisation is saved in \'%s\' file.' % loss_filename)

    fig = plt.figure()
    plt.plot(smbl_acc, label='train')
    plt.plot(val_smbl_acc, label='validation')
    plt.ylabel('Symbol accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    smbl_acc_filename = 'pics/smbl_acc' + ('' if n_iter == 0 else '_' + str(n_iter)) + '.png'
    fig.savefig(smbl_acc_filename)
    plt.close(fig)
    log('Symbol accuracy visualisation is saved in \'%s\' file.' % smbl_acc_filename)

    fig = plt.figure()
    plt.plot(sq_acc, label='train')
    plt.plot(val_sq_acc, label='validation')
    plt.ylabel('Sequence accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    sq_acc_filename = 'pics/sq_acc' + ('' if n_iter == 0 else '_' + str(n_iter)) + '.png'
    fig.savefig(sq_acc_filename)
    plt.close(fig)
    log('Sequence accuracy visualisation is saved in \'%s\' file.' % sq_acc_filename)
