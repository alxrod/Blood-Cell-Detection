import os
import numpy as np
from PIL import Image
import pickle

def convPklImg(path):
	for pkl_path in os.listdir(path):
		with open(path+pkl_path, "rb") as f:
			p = np.array(pickle.load(f))
			p = (p * 255).astype(np.uint8)
			img = Image.fromarray(np.uint8(p))
			img.save(path.replace("processed_pkls", "processed_images")+pkl_path.replace(".pkl",".jpg"))

convPklImg("../dataset/processed_pkls/train/")
