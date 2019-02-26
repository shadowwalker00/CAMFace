import numpy as np
import pickle
import os
import pandas as pd
import random
import argparse
import xlrd

def loadGender():
    face_rootpath = "/home/ghao/vizImpression/datasets/face_impression"
    whole_set = pd.read_pickle(os.path.join(face_rootpath, "dataset.pickle"))    
    gender_info = xlrd.open_workbook(os.path.join(face_rootpath, "demographic-others-labels.xlsx"))
    gender_table = gender_info.sheets()[1]
    nrows = gender_table.nrows
    filename_list = []
    gender_list = []
    for i in range(1, nrows):
        name, suffix = gender_table.row_values(i)[0].split(".")
        first_name = name+"."+suffix
        second_name = name + "_flip." + suffix
        filename_list.append(first_name)
        filename_list.append(second_name)
        gender_list.append(gender_table.row_values(i)[14])
        gender_list.append(gender_table.row_values(i)[14])
    dataset = pd.DataFrame({'image_path': filename_list})
    dataset["gender"] = gender_list
    dataset.to_pickle(os.path.join(face_rootpath, "gender.pickle"))
    male_index = np.where(np.array(gender_list)==1)
    female_index = np.where(np.array(gender_list) == 0)
    whole_set_male = whole_set.loc[male_index]
    whole_set_female = whole_set.loc[female_index]
    train_male = whole_set_male[:2028]
    test_male = whole_set_male[2028:]
    train_female = whole_set_female[:1524]
    test_female = whole_set_female[1524:]
    train_male.to_pickle(os.path.join(face_rootpath, "train_male.pickle"))
    test_male.to_pickle(os.path.join(face_rootpath, "test_male.pickle"))
    train_female.to_pickle(os.path.join(face_rootpath, "train_female.pickle"))
    test_female.to_pickle(os.path.join(face_rootpath, "test_female.pickle"))


def createTrain():
    face_rootpath = "/home/ghao/vizImpression/datasets/face_impression"
    data = xlrd.open_workbook(os.path.join(face_rootpath, 'psychology-attributes.xlsx'))
    table = data.sheets()[1]
    nrows = table.nrows
    #print(table.row_values(0))
    raw_data = []
    filename_list = []
    for i in range(1, nrows):
        filename_list.append(table.row_values(i)[0])
        name, suffix = table.row_values(i)[0].split(".")
        newname = name + "_flip." + suffix
        filename_list.append(newname)
        useful = np.array(table.row_values(i)[2:5] + table.row_values(i)[7:17] +
                          table.row_values(i)[20:30] + table.row_values(i)[32:44] + table.row_values(i)[47:])
        raw_data.append(useful)
        raw_data.append(useful)
    dataset = pd.DataFrame({'image_path': filename_list})
    dataset["label"] = raw_data
    trainset = dataset[:4000]
    testset = dataset[4000:]
    dataset.to_pickle(os.path.join(face_rootpath, "dataset.pickle"))
    trainset.to_pickle(os.path.join(face_rootpath, "train.pickle"))
    testset.to_pickle(os.path.join(face_rootpath, "test.pickle"))
    print("========================================")
    #print("MIT2kFace Dataset has already created in {}".format())
    print("Train Size: {}".format(len(trainset)))
    print("Test Size: {}".format(len(testset)))
    print("========================================")

if __name__ == "__main__":
    loadGender()
    #createTrain()
