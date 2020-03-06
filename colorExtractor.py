# USAGE: python colorExtractor.py --image dataset/starbucks-logo-vector.png

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-i",'--image',required=True,
    help = "Path to image")
args = vars(ap.parse_args())

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def HEX2RGB(color):
    a = list(color)
    a[0] = 'x'
    a.insert(0,'0')
    s = ""
    s.join(a)
    return s

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_colors(image, number_of_colors, show_chart):

    modified_image = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_

    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    values = []
    for i in counts.values():
        values.append(i)

    # if (show_chart):
    #     plt.figure(figsize = (8,6))
    #     plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    return hex_colors, values, rgb_colors

def read_csv():
    color_csv = pd.read_csv('color.csv')
    r = color_csv['_Red'].values
    g = color_csv['_Green'].values
    b = color_csv['_Blue'].values
    rgb = []
    for i in range(r.shape[0]):
        temp = []
        temp.append(r[i])
        temp.append(g[i])
        temp.append(b[i])
        # print(temp, len(temp))
        rgb.append(temp)
    name = list(color_csv['_Title'].values)
    return rgb, name

available_colors, color_names = read_csv()

colors, values, rgb_colors = get_colors(get_image(args['image']), 10, True)

def closest(colors,color):
    colors = np.array(colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))
    
    if len(index_of_smallest[0]) != 1:
        index = index_of_smallest[0][0]
    else:
        index = index_of_smallest[0]

    color_name = color_names[np.asscalar(index)]

    return color_name 

def color_content(values, colors, rgb_colors):
    tot = 0
    for i in values:
        tot+=i
    color_dict = {}
    for i in range(len(colors)):
        color = closest(available_colors,rgb_colors[i])
        color_dict[color] = 0
    for i in range(len(colors)):
        color = closest(available_colors,rgb_colors[i])
        color_dict[color] += (values[i]/tot)*100
    return color_dict

print(color_content(values, colors, rgb_colors))


