import os
import numpy as np
from PIL import Image
import pickle
from sklearn.preprocessing import OneHotEncoder

def load_folder(path):
	images = []
	labels = []
	for pkl_path in os.listdir(path):
	#	with open(path+pkl_path, "rb") as f:
	#		p = pickle.load(f)
		images.append(path+pkl_path)
		labels.append(pkl_path.split("_")[0])
	unique_labels = []
	for i in labels:
		if i not in unique_labels:
			unique_labels.append(i)
	lbs_to_ints = {}
	for l in unique_labels:
		lbs_to_ints[l] = unique_labels.index(l)
	encoded_labels = []
	for l in labels:
		encoded_labels.append(lbs_to_ints[l])

	encoded_labels = np.reshape(encoded_labels, (-1,1))	
	
	enc = OneHotEncoder()
	train_labels = enc.fit_transform(encoded_labels).toarray()

	return train_labels, images, lbs_to_ints

# labels, images, lti = load_folder("dataset/processed_images/train/")
# print(lti)
# print(labels)
