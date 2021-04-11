import cv2
import numpy as np
import os
from model import dual_net
import keras
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import math

def get_data_rope_run(runNumber):

    x_data = [] # x -> (n_samples, 2, 240, 240, 3)
    y_data = [] # y -> (n_samples, 4)
    img_list = []

    run_folder = os.path.join("ball","run{}".format(runNumber))
    action_file = os.path.join(run_folder,"actions.npy")
    actions = np.load(action_file)

    count = 0
    for file_name in os.listdir(run_folder):
        if file_name != "actions.npy":
            img_path = os.path.join(run_folder,file_name)
            img = cv2.imread(img_path)
            img_list.append(img)
    
    for i in range(len(img_list)-1):
        x_data.append([img_list[i],img_list[i+1]])
        y_data.append(actions[i][:4]) #Dont take last index

    return np.array(x_data), np.array(y_data)

def get_rope_data(runs):

    x_data = []
    y_data = []

    for i in range(0,runs):
        runNum = str(i).zfill(2)
        x_temp, y_temp = get_data_rope_run(runNum)
        if i == 0:
            x_data = x_temp
            y_data = y_temp
        else:
            for j in range(0,len(x_temp)):
                x_data.append(x_temp[j])
                y_data.append(y_temp[j])
        print("Retrived Run{}".format(runNum))
    
    #x_data = np.array(x_data)
    #y_data = np.array(y_data)

    rng_state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(rng_state)
    np.random.shuffle(y_data)
    
    return x_data,y_data

def format_data(x, y):

    batch_x = np.zeros((2, len(x), 240, 240, 3))
    batch_y = np.zeros((len(y), 4))

    batch_start_index = 0
    batch_stop_index = len(x)-1

    # get the samples in the batch
    counter = 0
    while batch_start_index < batch_stop_index:
    	batch_x[0][counter] = x[batch_start_index][0]
    	batch_x[1][counter] = x[batch_start_index][1]
    	batch_y[counter] = y[batch_start_index]
    	counter = counter + 1
    	batch_start_index = batch_start_index + 1

    return batch_x, batch_y

def getDistance(p1,p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def getNewPoint(point,angle,dist):
	new_x = point[0] + dist*math.cos(angle)
	new_y = point[1] + dist*math.cos(angle)
	return [new_x,new_y]

def test(x_data,y_data,model):

	num_test_samples = len(x_data)

	preds = model.predict([x_data[0],x_data[1]])

	startPointError = []
	endPointError = []

	for i,pred in enumerate(preds):
		predStart = pred[:2]
		trueStart = y_data[i][:2]
		startPointError.append(getDistance(predStart,trueStart))

		predEnd = getNewPoint(pred[:2],pred[2],pred[3])
		trueEnd = getNewPoint(y_data[i][:2],y_data[i][2],y_data[i][3])
		endPointError.append(getDistance(predEnd,trueEnd))

	averageStartPointError = sum(startPointError)/len(startPointError)
	averageEndPointError = sum(endPointError)/len(endPointError)

	print("The Average Start Point Error is {} pixels".format(averageStartPointError))
	print("The Average End Point Error is {} pixels".format(averageEndPointError))

	return startPointError,endPointError

def main():

    rope_runs = 1

    print("Getting Data...")
    X,y = get_rope_data(rope_runs)
    X,y = format_data(X,y)

    model = keras.models.load_model("model_ball.h5")

    st, en = test(X,y,model)

    x = []
    y = []
    for i in range(len(st)):
        if st[i]<20:
            x.append(st[i])
        if en[i]<20:
            y.append(en[i])

    print(len(x)/len(st))
    print(len(y)/len(en))
    

if __name__ == '__main__':
    main()

