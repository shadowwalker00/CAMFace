import numpy as np
import pickle
import os
import pandas as pd
import random
import argparse
import xlrd


class Dataset:
	def __init__(self):
		self.caltech256_rootpath = "/home/ghao/vizImpression/datasets/caltech256/"
		self.caltech256_path = "/home/ghao/vizImpression/datasets/caltech256/256_ObjectCategories"
		self.face_rootpath = "/home/ghao/vizImpression/datasets/face_impression"
		self.face_path = "/home/ghao/vizImpression/datasets/face_impression"
	
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
			print("Caltech256 Dataset has already created in {}".format(self.caltech256_rootpath))
			print("Train Size: {}".format(len(trainset)))
			print("Test Size: {}".format(len(testset)))
			print("========================================")
		elif datasetName == "face":
			if not os.path.exists(trainset_path):
				data = xlrd.open_workbook(os.path.join(self.face_rootpath,'psychology-attributes.xlsx'))
				table = data.sheets()[1]				
				nrows = table.nrows
				print(table.row_values(0))
				raw_data = []
				filename_list = []
				for i in range(1,nrows):
					filename_list.append(table.row_values(i)[0])
					newname = data_augmentation(table.row_values(i)[0])
					filename_list.append(newname)
					useful = np.array(table.row_values(i)[2:5]+table.row_values(i)[7:17]+
							table.row_values(i)[20:30]+table.row_values(i)[32:44]+table.row_values(i)[47:])
					raw_data.append(useful)
					raw_data.append(useful)
				dataset = pd.DataFrame({'image_path': filename_list})
				dataset["label"] = raw_data
				trainset = dataset[:4000]
				testset = dataset[4000:]
				trainset.to_pickle(os.path.join(self.face_rootpath,"train.pickle"))
				testset.to_pickle(os.path.join(self.face_rootpath,"test.pickle"))	
				print("========================================")
				print("MIT2kFace Dataset has already created in {}".format(self.face_rootpath))
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
		else datasetname == "face":
			if os.path.exists(os.path.join(self.face_rootpath,"train.pickle")):
				trainset = pd.read_pickle(os.path.join(self.face_rootpath,"train.pickle"))
			if os.path.exists(os.path.join(self.face_rootpath,"test.pickle")):
				testset = pd.read_pickle(os.path.join(self.face_rootpath,"test.pickle"))
		return trainset,testset
		

if __name__=="__main__":
    #create dataset
    parser = argparse.ArgumentParser()
    parser.description='Dataset parser'
    parser.add_argument('--name', help="name of dataset")
    allPara = parser.parse_args()
    data = Dataset()
    data.createDataSet(allPara.name)