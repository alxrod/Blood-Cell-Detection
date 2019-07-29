from PIL import Image
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import pickle

# Final Image
def convert_image(path):
    starter_img = Image.open(path)
    array_img = np.array(starter_img)
    blue_image = array_img[:,:,1]
    

    elim_black = lambda x : 255 if x < 40 else x
    vfunc = np.vectorize(elim_black)
    whitened_image = vfunc(blue_image).astype("uint8")
    
    thresh=whitened_image.min()*1.4
    xS = []
    yS = []
    while len(xS) < 64:
        elim_white = lambda x : 255 if x > thresh else x
        vfunc2 = np.vectorize(elim_white)
        center_image = vfunc2(whitened_image).astype("uint8")

        for x in range(center_image.shape[0]):
            for y in range(center_image.shape[1]):
                if center_image[x][y] < 200 and center_image[x][y] > 60:
                    xS.append(x)
                    yS.append(y)
        cX = np.mean(xS)
        cY = np.mean(yS)
        if np.isnan(cX) == False:
            center = (int(cX), int(cY))
            padded = np.pad(array_img, ((56, 56), (56, 56), (0,0)), 'constant', constant_values=0)
#             return padded
            final_image = padded[center[0]:center[0]+112,center[1]:center[1]+112,:]
        else:
            xS = []
            yS = []
            thresh+=thresh*0.2
    return final_image

# Normalization strategy:
def normalize_img(im_ar):
#   Gaussian Blur to start with 3x3 kernel
    im_ar = cv2.GaussianBlur(im_ar,(3,3),0)
    xMin = im_ar.min()
    xMax = im_ar.max()
    im_ar = (im_ar-xMin)/(xMax-xMin)
    return im_ar

# So I'm going to populate my own dataframe
img_paths = []
img_types = []
train_dir = "../dataset/images/TRAIN/"
cell_types = ["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]

for c_type in cell_types:
    path = train_dir + c_type
    for p in os.listdir(path):
        img_paths.append(path+"/"+p)
        img_types.append(c_type)
labels = pd.DataFrame({"cell_type": img_types,"img_path":img_paths})

# Testing Generalization:

# sample = labels.sample(n=3, random_state=k)

total = len(labels.values)
for i, row in enumerate(labels.values):
	print("Progress: " + str((i/total)*100) + "%")
	cell_type, path = row
	im = normalize_img(convert_image(path).astype("float"))
	filename = path.split("/")[-1].split(".jpeg")[0]
	new_path = "../dataset/processed_images/"+cell_type+filename+".pkl"
	with open(new_path, "wb") as f:
		pickle.dump(im, f)



    