import os


def get_splitted_processes(directories, in_suffix):
    processes = []
    base_dir = 'data/'
    for directory in directories:
        work_dir = base_dir + directory + in_suffix
        for sub_dir in os.listdir(work_dir):
            file_dir = work_dir + '/' + sub_dir
            # print('Directory', file_dir, 'contains', len(os.listdir(file_dir)), 'processes')
            for process in os.listdir(file_dir):
                with open(file_dir + '/' + process, 'r') as file:
                    content = file.read().strip()
                    processes.append(content.split())
    return processes


def get_processes_grouped_by_name(directories, in_suffix):
    processes = {}
    base_dir = 'data/'
    for directory in directories:
        work_dir = base_dir + directory + in_suffix
        for sub_dir in os.listdir(work_dir):
            file_dir = work_dir + '/' + sub_dir
            for process in os.listdir(file_dir):
                process_name = sub_dir + '/' + process[:process.find('_')]
                whole_process = processes.get(process_name, [])
                with open(file_dir + '/' + process, 'r') as file:
                    content = file.read().strip()
                    whole_process += content.split()
                processes[process_name] = whole_process
    return processes
