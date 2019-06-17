import os
import operator
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from lxml import etree

from dictionary.Constants import colors
from dictionary.DictionaryZipper import zip_event
from model_1_clf.ToTextConverter import get_new_word


def draw_rectangles(events):
    global color_pos
    rectangles = []
    x, y = 0, 0
    for event in events:
        if event not in colors_map:
            colors_map[event] = color_pos
            color_pos += 1
        rectangles.append(patches.Rectangle((x, y), width=rec_size, height=rec_size, linewidth=1,
                                            color=colors[colors_map[event]], fill='True'))
        x += rec_size
        if x % pic_size == 0:
            x = 0
            y += rec_size
    return rectangles


def draw_processes(directories, zip_event, get_new_word):
    base_dir = 'data/'
    for directory in directories:
        work_dir = base_dir + directory + '_splitted'
        for sub_dir in os.listdir(work_dir):
            file_dir = work_dir + '/' + sub_dir
            for original_process_file in os.listdir(file_dir):
                process_name = original_process_file[:-4]
                process = []
                context = etree.iterparse(file_dir + '/' + original_process_file, events=('end',), tag=('event',))
                for _, elem in context:
                    parameters = {}
                    for child in elem.iterchildren(tag=('Operation', 'Path', 'Result')):
                        parameters[child.tag] = child.text
                    zip_event(parameters)
                    process.append(get_new_word(parameters))
                start = 0
                iter = 0
                while True:
                    iter += 1
                    end = start + pic_size * pic_size // (rec_size * rec_size)
                    img = Image.new('RGB', (pic_size, pic_size), color='white')
                    fig, ax = plt.subplots(1)
                    print(fig.get_dpi())
                    ax.imshow(np.array(img, dtype=np.uint8))
                    rectangles = draw_rectangles(process[start:end])
                    for rectangle in rectangles:
                        ax.add_patch(rectangle)
                    if not os.path.exists('/'.join(['pics', directory, sub_dir])):
                        os.makedirs('/'.join(['pics', directory, sub_dir]))
                    fig.savefig('pics/%s/%s/%s_%s.png' % (directory, sub_dir, process_name, str(iter)), dpi=300)
                    plt.close(fig)
                    start = end
                    if start >= len(process):
                        break


colors_map = {}
color_pos = 0
pic_size = 600
rec_size = 10

if __name__ == '__main__':
    draw_processes(['ransomware', 'clean_win7', 'clean_win10'], zip_event, get_new_word)
