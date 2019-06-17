class Event:
    def __init__(self, operation, path, result):
        self.__operation = operation
        self.__path = path
        self.__result = result

    def get_operation(self):
        return self.__operation

    def get_path(self):
        return self.__path

    def get_result(self):
        return self.__result

    def get_all_parameters(self):
        params = {'Operation': self.__operation, 'Path': self.__path, 'Result': self.__result}
        return params
