import numpy as np
import pickle
import os
import pandas as pd
import random
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

with open('/home/ghao/vizImpression/out/loss.pkl','rb') as f:
	loss = pickle.load(f)
	plt.plot(range(len(loss)),loss)
	plt.savefig("./loss_train_whole.png")
	print("done")

