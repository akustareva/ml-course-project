from utils.GetProcesses import *
from sklearn.model_selection import train_test_split


def log(text, filename='res.txt', end='\n', console=True, file=True):
    if console:
        print(text, end=end)
    if file:
        out = open(filename, 'a')
        out.write(text + end)
        out.close()


def train_validation_test_split(clean_dirs, infected_dirs, suffix, log):
    clean_processes = get_splitted_processes(clean_dirs, suffix)
    infected_processes = get_splitted_processes(infected_dirs, suffix)
    normal_train, normal_test = train_test_split(clean_processes, test_size=0.035, shuffle=False)
    normal_validation, normal_test = train_test_split(normal_test, test_size=0.5, shuffle=False)
    infected_validation, infected_test = train_test_split(infected_processes, test_size=0.5, shuffle=True)
    log('Count of train processes: %d' % len(normal_train))
    log('Count of normal validation processes: %d' % len(normal_validation))
    log('Count of normal test processes: %d' % len(normal_test))
    log('Count of anomalous validation processes: %d' % len(infected_validation))
    log('Count of anomalous test processes: %d' % len(infected_test))
    return normal_train, normal_validation, normal_test, infected_validation, infected_test
