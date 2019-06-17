import os
import datetime
import operator
import matplotlib.pyplot as plt

from lxml import etree
from sklearn.externals import joblib
from timeit import default_timer as timer

from dictionary.DictionaryZipper import *


def get_files_from_dir(d):
    res = []
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith(".xml"):
                res.append("/".join([root, file]))
    return res


def add_statistics(parameters):
    operations[parameters['Operation']] = operations.get(parameters['Operation'], 0) + 1
    results[parameters['Result']] = results.get(parameters['Result'], 0) + 1
    paths[parameters['Path']] = paths.get(parameters['Path'], 0) + 1


operations = {}
results = {}
paths = {}

if __name__ == '__main__':
    basedir = 'data/'
    total_start = timer()
    for work_dir in ['clean_win10_splitted/', 'clean_win7_splitted/', 'ransomware_splitted/']:
        in_dir = basedir + work_dir
        out_dir = in_dir[:-9] + 'zipped/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print('Start', in_dir, 'directory')
        files = get_files_from_dir(in_dir)
        for file in files:
            start = timer()
            file_parts = file.split('/')
            out_filename = '/'.join(file_parts[2:])
            print('  File', out_filename)
            if not os.path.exists(out_dir + out_filename.split('/')[0]):
                os.makedirs(out_dir + out_filename.split('/')[0])
            out_file = open(out_dir + out_filename, 'wb')
            context = etree.iterparse(file, events=('end',), tag=('event',))
            process = etree.Element('process')
            for _, elem in context:
                event = etree.SubElement(process, 'event')
                parameters = {}
                for child in elem.iterchildren(tag=('Operation', 'Path', 'Result')):
                    parameters[child.tag] = child.text
                zip_event(parameters)
                add_statistics(parameters)
                for k, v in sorted(parameters.items(), key=operator.itemgetter(0)):
                    info = etree.SubElement(event, k)
                    info.text = v
                elem.clear()
            tree = etree.ElementTree(process)
            tree.write(out_file, pretty_print=True, xml_declaration=True, encoding="utf-8")
            out_file.close()
            end = timer()
            print('  Finished. Time:', str(datetime.timedelta(seconds=end - start)))
    print('Total execution time:', str(datetime.timedelta(seconds=timer() - total_start)))

    print('Dumping statistics...')
    joblib.dump((operations, results, paths), "dumps/zipped_statistics.pkl")
    sorted_paths = sorted(paths.items(), key=operator.itemgetter(1))
    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    sorted_operations = sorted(operations.items(), key=operator.itemgetter(1))

    plt.figure(figsize=(10, 5))
    plt.xticks(range(len(sorted_paths)), [p[0] for p in sorted_paths])
    plt.plot(range(len(sorted_paths)), [p[1] for p in sorted_paths], 'ro--')
    plt.yscale('log')
    plt.show()

    plt.figure(figsize=(13, 5))
    plt.xticks(range(len(sorted_results)), [p[0] for p in sorted_results], rotation=20)
    plt.plot(range(len(sorted_results)), [p[1] for p in sorted_results], 'ro--')
    plt.yscale('log')
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.xticks(range(len(sorted_operations)), [p[0] for p in sorted_operations], rotation=85)
    plt.plot(range(len(sorted_operations)), [p[1] for p in sorted_operations], 'ro--')
    plt.yscale('log')
    plt.show()
