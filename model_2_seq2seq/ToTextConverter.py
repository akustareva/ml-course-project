from dictionary.Dictionary import *
from collections import defaultdict
from utils.ToTextConverter import convert
from dictionary.DictionaryZipper import zip_event


def one_hot_feature_encoding(pos, cnt):
    lst = [0] * cnt
    lst[pos] = 1
    return ''.join(map(str, lst))


def get_new_word(parameters):
    operation_id = operation_type_map[parameters['Operation']]
    result_id = result_type_map[parameters['Result']]
    path_id = path_type_map[parameters['Path']]
    # return one_hot_feature_encoding(operation_id, operations_count) + \
    #        one_hot_feature_encoding(result_id, results_count) + one_hot_feature_encoding(path_id, paths_count)
    event = '_'.join(map(str, [operation_id, path_id, result_id]))
    if event not in events:
        events[event] = str(len(events))
    return events[event]


events = defaultdict(int)

if __name__ == '__main__':
    convert(['clean', 'infected'], 'model_2_text', zip_event, get_new_word)
    print(len(events))
    print(events)
