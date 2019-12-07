import os

train = 0.75
dataset_path = "../dataset/processed_images/"
# test = the rest

def split_data(train_perc):
	files = os.listdir(dataset_path)
	if "test" in files and "train" in files:
		print("You have already split the data")
		return
	else:
		os.mkdir(dataset_path+"test")
		os.mkdir(dataset_path+"train")
		data_files = [f for f in files if ".pkl" in f]
		num_of_train = len(data_files)*train_perc
		count = 0
		for f in data_files:
			if count < num_of_train:
				os.rename(dataset_path+f, dataset_path+"train/"+f)
			else:
				os.rename(dataset_path+f, dataset_path+"test/"+f)
			count+=1

if __name__ == '__main__':
	split_data(0.75)
