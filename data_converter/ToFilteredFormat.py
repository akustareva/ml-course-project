import os
import datetime

from lxml import etree
from timeit import default_timer as timer


if __name__ == '__main__':
    basedir = 'data/'
    for work_dir in ['clean_win10/', 'clean_win7/', 'ransomware/']:
        in_dir = basedir + work_dir
        out_dir = in_dir[:-1] + '_filtered/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print('Start', in_dir, 'directory')
        for file in os.listdir(in_dir):
            start = timer()
            infile = in_dir + file
            outfile = open(out_dir + file, 'wb')
            context = etree.iterparse(infile, events=('end',), tag=('process', 'event'))
            procmon = etree.Element('procmon')
            process_list = etree.SubElement(procmon, 'processlist')
            event_list = etree.SubElement(procmon, 'eventlist')
            for _, elem in context:
                if elem.tag == 'process':
                    process = etree.SubElement(process_list, 'process')
                    for child in elem.iterchildren(tag=('ProcessIndex', 'ProcessName')):
                        sub_el = etree.SubElement(process, child.tag)
                        sub_el.text = child.text
                elif elem.tag == 'event':
                    event = etree.SubElement(event_list, 'event')
                    for child in elem.iterchildren(tag=('ProcessIndex', 'Process_Name', 'Operation', 'Path', 'Result')):
                        sub_el = etree.SubElement(event, child.tag)
                        sub_el.text = child.text
                elem.clear()
            tree = etree.ElementTree(procmon)
            tree.write(outfile, pretty_print=True, xml_declaration=True, encoding="utf-8")
            outfile.close()
            print('  ' + file + ': ', str(datetime.timedelta(seconds=timer() - start)))
