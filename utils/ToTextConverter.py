import os
import datetime

from lxml import etree
from timeit import default_timer as timer


def convert(directories, out_suffix, zip_event, get_new_word):
    basedir = 'data/'
    for work_dir in directories:
        in_dir = basedir + work_dir + '_splitted'
        print('Start %s directory' % in_dir)
        start = timer()
        out_dir = basedir + work_dir + '_' + out_suffix
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for sub_dir in os.listdir(in_dir):
            in_file_dir = in_dir + '/' + sub_dir
            for file in os.listdir(in_file_dir):
                out_file_dir = out_dir + '/' + sub_dir
                filename = out_file_dir + '/' + file[:-4] + '.txt'
                if not os.path.exists(out_file_dir):
                    os.makedirs(out_file_dir)
                out_file = open(filename, 'w')
                context = etree.iterparse(in_file_dir + '/' + file, events=('end',), tag='event')
                for _, event in context:
                    parameters = {}
                    for child in event.iterchildren(tag=('Operation', 'Path', 'Result')):
                        parameters[child.tag] = child.text
                    zip_event(parameters)
                    new_word = get_new_word(parameters)
                    if new_word != '':
                        out_file.write(new_word + ' ')
                    event.clear()
                out_file.close()
        end = timer()
        print('Finished. Time:', str(datetime.timedelta(seconds=end - start)))
