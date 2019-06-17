import os
import datetime
import operator

from lxml import etree
from collections import defaultdict
from timeit import default_timer as timer

from data_converter.entity.Event import *

MIN_EVENTS_CNT = 30

if __name__ == '__main__':
    basedir = 'data/'
    for work_dir in ['clean_win10_filtered/', 'clean_win7_filtered/', 'ransomware_filtered/']:
        in_dir = basedir + work_dir
        out_dir = in_dir[:-9] + 'splitted/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print('Start', in_dir, 'directory')
        for file in os.listdir(in_dir):
            start = timer()
            files_cnt = 0
            print('  File', file)
            infile = in_dir + file
            context = etree.iterparse(infile, events=('end',), tag=('process', 'event'))
            processes = {}
            process_events = defaultdict(list)
            for _, elem in context:
                if elem.tag == 'process':
                    params = {}
                    for child in elem.iterchildren(tag=('ProcessIndex', 'ProcessName')):
                        params[child.tag] = child.text
                    processes[params['ProcessIndex']] = params['ProcessName']
                else:
                    params = {}
                    for child in elem.iterchildren(tag=('ProcessIndex', 'Process_Name', 'Operation', 'Path', 'Result')):
                        params[child.tag] = child.text
                    process_index = params['ProcessIndex']
                    if process_index not in processes.keys():
                        print('    UNKNOWN PI', process_index)
                    elif params['Process_Name'] != processes[process_index]:
                        print('    DIFFERENT PROCESS NAMES', params['Process_Name'], processes[process_index],
                              'FOR PI', process_index)
                    else:
                        process_events[process_index].append(Event(params['Operation'], params['Path'], params['Result']))
                elem.clear()
            names = {}
            subdir = file[:-4] + '/'
            if not os.path.exists(out_dir + subdir):
                os.makedirs(out_dir + subdir)
            for process_index, events_list in process_events.items():
                if len(events_list) < MIN_EVENTS_CNT:
                    continue
                files_cnt += 1
                name = processes[process_index]
                name_cnt = names.get(name, 0) + 1
                names[name] = name_cnt
                outfile_name = out_dir + subdir + name + '_' + str(name_cnt) + '.xml'
                outfile = open(outfile_name, 'wb')
                process = etree.Element('process')
                for event in events_list:
                    event_params = event.get_all_parameters()
                    e = etree.SubElement(process, 'event')
                    for k, v in sorted(event_params.items(), key=operator.itemgetter(0)):
                        info = etree.SubElement(e, k)
                        info.text = v
                tree = etree.ElementTree(process)
                tree.write(outfile, pretty_print=True, xml_declaration=True, encoding="utf-8")
                outfile.close()
            end = timer()
            print('    Count of processes:', len(processes))
            print('    Count of files:', files_cnt)
            print('  Finished. Time:', str(datetime.timedelta(seconds=end - start)))
