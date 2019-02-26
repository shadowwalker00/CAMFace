import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, localcontext
from datetime import datetime
import pytz
import time
import os
import pandas as pd
import numpy as np
import pickle as Pickle
import tensorflow as tf
from util import *
from detector import Detector
import argparse

def diffFiles(file1, file2):
	with open(file1) as f:
		data1 = f.readlines()
	with open(file2) as f:
		data2 = f.readlines()
	out1 = []
	for i in data1:
		temp = i.strip().split(" ")
		newtemp = [item for item in temp if item != '']
		out1.append((newtemp[0],newtemp[1],newtemp[2]))
	out1 = sorted(out1, key=lambda x : x[0])
	print(out1)


	out2= []
	for i in data2:
		temp = i.strip().split(" ")
		newtemp = [item for item in temp if item != '']
		out2.append((newtemp[0],newtemp[1],newtemp[2]))
	out2 = sorted(out2, key=lambda x : x[0])
	print(out2)


	with open("./out/diff.txt","w+") as f:		
		for i in range(len(out1)):
			f.write(out1[i][0].ljust(20)+"&"+str(out1[i][1])+"&"+str(out2[i][1])+"&"+str(float(out1[i][1])-float(out2[i][1]))+"\\\\"+"\n")
	






def plotScatter():
	with open("./out/best.txt") as f:
		data1 = f.readlines()
	baseline = [("happy",0.84),("unhappy",0.75),("friendly",0.78),("unfriendly",0.72),("sociable",0.74),("introverted",0.50),("attractive",0.72),("unattractive",0.62),("kind",0.72),("mean",0.69),("caring",0.72),("cold",0.71),("trustworthy",0.62),
	("untrustworthy",0.60),("responsible",0.58),("irresponsible",0.55),("confident",0.55),("uncertain",0.45),("humble",0.55),("egotistic",0.52),
	            ("emotStable",0.53),("emotUnstable",0.50),("normal",0.49),("weird",0.52),("intelligent",0.49),("unintelligent",0.43),("interesting",0.42),("boring",0.39),
	            ("calm",0.41),("aggressive",0.65),("emotional",0.33),("unemotional",0.56),("memorable",0.30),("forgettable",0.27),("typical",0.28),("atypical",0.24),("common",0.25),
	            ("uncommon",0.27),("familiar",0.24),("unfamiliar",0.18)]
	baseline = sorted(baseline,key=lambda x:x[0])
	print(baseline)
	myModel = []
	for i in data1:
		temp = i.strip().split("\t")
		newtemp = [item for item in temp if item != '']
		myModel.append((newtemp[0],newtemp[1]))
	myModel = sorted(myModel,key=lambda x:x[0])
	x = []
	for i in range(len(baseline)):
		print(baseline[i][0],baseline[i][1],myModel[i][0],myModel[i][1])
		x.append(myModel[i][1])
	index = np.argsort(np.array(x))
	dataX = []
	dataY = []
	label = []
	for i in index:
		with localcontext() as ctx:
			ctx.prec = 2
			temp= Decimal(str(myModel[i][1]))
			dataX.append(float(temp))
		dataY.append(baseline[i][1])	
		label.append(baseline[i][0])

	#fig, axs = plt.figure()
	plt.scatter(dataX,dataY)
	for i, xy in enumerate(zip(dataX, dataY)):
		plt.annotate("%s" % label[i][:4], xy=xy, xytext=(5, 0), textcoords='offset points')
	plt.title("Correlation")
	plt.xlabel("Model Correlation (with humans)")
	plt.ylabel("Human Consensun (with each other)")
	plt.grid()
	plt.xlim((0.25, 0.85))
	plt.ylim((0.25, 0.85))
	#plt.axes().set_aspect('equal', 'datalim')
	plt.gca().set_aspect('equal', adjustable='box')
	#axs.set_aspect('equal', 'box')
	plt.savefig("./correlation.png")

def balabal():
	root_path = '/home/ghao/vizImpression'
	image_path = os.path.join(root_path,'datasets/face_impression/images')
	trainset_path = os.path.join(root_path,'datasets/face_impression/train_female.pickle')
	weight_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/caffe_layers_value.pickle')
	model_path = os.path.join(root_path,'trained_models/VGG/face')
	pretrained_path = os.path.join(root_path,'trained_models/pretrained_weight/VGG/')
	trainset = pd.read_pickle(trainset_path)
	print ('Read from disk: trainset(Size):{}'.format(len(trainset)))
if __name__=="__main__":
	diffFiles("./out/FemaleModel.txt", "./out/AllModel.txt")
	#balabal()