import numpy as np
import pickle
import os
import pandas as pd
import random

class Dataset:
	def __init__(self):
		self.caltech256_rootpath = "/home/ghao/vizImpression/datasets/caltech256/"
		self.caltech256_path = "/home/ghao/vizImpression/datasets/caltech256/256_ObjectCategories"
	
	def createDataSet(self, datasetName):
		"""
		create dataset with specific name and store train and test set into pickle file
		datasetName: name of dataset
		"""
		if datasetName == "caltech":
			filelist = []
			labelist = []
			for dirname in os.listdir(self.caltech256_path):
				classLabel,classname = os.path.splitext(dirname)
				classLabel = classLabel.strip()
				classname = classname.strip()
				if classLabel.isdigit():
					for filename in os.listdir(os.path.join(self.caltech256_path,dirname)):
						filelist.append(os.path.join(self.caltech256_path,dirname,filename))
						labelist.append(int(classLabel)-1)

			#make a pandas to store dataset		
			dataDict = {"file":filelist,"label":labelist}
			dataset = pd.DataFrame(dataDict)

			#split dataset into 80% size and 20% size
			li=list(range(len(dataset)))
			random.shuffle(li)
			trainsize = int(len(li) *0.8)
			testsize = len(li)- trainsize
			trainset = dataset.loc[li[:trainsize]]
			testset = dataset.loc[li[trainsize:]]
			#store into file
			trainset.to_pickle(os.path.join(self.caltech256_rootpath,"train.pickle"))
			testset.to_pickle(os.path.join(self.caltech256_rootpath,"test.pickle"))
			print("========================================")
			print("Caltech257 Dataset has already created.")
			print("Train Size: {}".format(len(trainset)))
			print("Test Size: {}".format(len(testset)))
			print("========================================")

	def getDataset(self,datasetname):
		"""
		load specific dataset from from disk
		datasetName: name of dataset
		return: trainset and testset
		"""
		if datasetname == "caltech":
			if os.path.exists(os.path.join(self.caltech256_rootpath,"train.pickle")):
				trainset = pd.read_pickle(os.path.join(self.caltech256_rootpath,"train.pickle"))
			if os.path.exists(os.path.join(self.caltech256_rootpath,"test.pickle")):
				testset = pd.read_pickle(os.path.join(self.caltech256_rootpath,"test.pickle"))
		return trainset,testset
		

if __name__=="__main__":
	data = Dataset()
	data.createDataSet("caltech")
	trainset,testset = data.getDataset("caltech")
	print(trainset)
	print(testset)