from dictionary.Dictionary import *
from dictionary.DictionaryZipper import zip_event
from utils.ToTextConverter import convert


def get_new_word(parameters):
    operation_id = operation_type_map[parameters['Operation']]
    result_id = result_type_map[parameters['Result']]
    path_id = path_type_map[parameters['Path']]
    return '_'.join(map(str, [operation_id, path_id, result_id]))


if __name__ == '__main__':
    convert(['clean', 'infected'], 'model_1_text', zip_event, get_new_word)
