from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import  pathlib
import os 
import random
import pickle
import numpy as np

def load_image(img_path,label):
    image = pickle.load(open(img_path.numpy().decode(),"rb"))
    t_img = tf.convert_to_tensor(image, dtype=tf.float32)
    return t_img, label
def load_dataset():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    root = "../dataset/processed_images/"
    files = [root+f for f in os.listdir(root) if ".pkl" in f]

    random.shuffle(files)
    image_count = len(files)
    print(image_count)

    str_labels = [f.split("/")[-1].split("_")[0] for f in files]
    print(str_labels[:10])
    cell_types = ["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]
    for f in files:
        none = False
        for t in cell_types:
            if t in f:
                none = True
        if none == False:
            print("Unclassified data entry " + f )
            os.remove(f)
            files.remove(f)
    label_to_index = dict((name, index) for index, name in enumerate(cell_types))
    print(label_to_index[str_labels[0]])
    int_labels = [label_to_index[s] for s in str_labels]

    ds = tf.data.Dataset.from_tensor_slices((files, int_labels))
    image_label_ds = ds.map(lambda filename, label: tuple(tf.py_function(load_image, [filename, label], [tf.float32,label.dtype])) )

     # Batch the data!!!
   
    return image_label_ds


