import cv2
import numpy as np
import os
import math
from model_skip import dual_net
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import tensorflow as tf


def get_data_rope_run(runNumber):

    x_data = [] # x -> (n_samples, 2, 240, 240, 3)
    y_data = [] # y -> (n_samples, 4)
    img_list = []

    run_folder = os.path.join("rope","run{}".format(runNumber))
    action_file = os.path.join(run_folder,"actions.npy")
    actions = np.load(action_file)

    count = 0
    for file_name in os.listdir(run_folder):
        if file_name != "actions.npy":
            img_path = os.path.join(run_folder,file_name)
            img = cv2.imread(img_path)
            img_list.append(img)
    
    for i in range(len(img_list)-2):
        x_data.append([img_list[i],img_list[i+2]]) #Skip an image
        action1 = actions[i][:4] #Change if predicting all 8 actions or 1
        action2 = actions[i+1][:4]
        both_actions = np.concatenate([action1,action2])
        y_data.append(both_actions) #Dont take last index

    return x_data, y_data

def get_rope_data(runs):

    x_data = []
    y_data = []

    for i in range(3,runs):
        runNum = str(i).zfill(2)
        x_temp, y_temp = get_data_rope_run(runNum)
        if i == 3:
            x_data = x_temp
            y_data = y_temp
        else:
            for j in range(0,len(x_temp)):
                x_data.append(x_temp[j])
                y_data.append(y_temp[j])
        print("Retrived Run{}".format(runNum))

    rng_state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(rng_state)
    np.random.shuffle(y_data)
    
    return np.array(x_data),np.array(y_data)

def create_batch(batch_size, step, x_train,y_train):

    batch_x = np.zeros((2,batch_size,240,240,3))
    batch_y = np.zeros((batch_size,8)) #Change if predicting all 8 actions or 1

    batch_start_index = step * batch_size
    batch_stop_index = (step + 1) * batch_size - 1

    # get the samples in the batch
    counter = 0
    while batch_start_index < batch_stop_index:
    	batch_x[0][counter] = x_train[batch_start_index][0]
    	batch_x[1][counter] = x_train[batch_start_index][1]
    	batch_y[counter] = y_train[batch_start_index]
    	counter = counter + 1
    	batch_start_index = batch_start_index + 1

    return batch_x,batch_y



def train(x_train, y_train, x_validation, y_validation, epochs, batch_size):

	prev_validation_loss = 1000000 # max loss (used to save model with lowest validation loss)

	# initialize model and optimizer
	model = dual_net()
	optimizer= Adam(learning_rate=0.0001)

	# number of steps per epoch
	max_steps = int(len(x_train)/batch_size)

	print('Number of steps per epoch : ', max_steps)

	print("Training...")
	loss_list = []

	# training loop
	for epoch in range(1,epochs+1):

		epoch_loss = []
		print('____________________________________________________')
		print('Epoch: ', epoch)
		step = 0

		# loop through all batches. 1 batch/ step. each step updates the parameters after batch_size samples
		while step < max_steps:

			# get batch
			batch_x,batch_y = create_batch(batch_size, step, x_train, y_train)

			with tf.GradientTape() as tape:
				y_pred = model([batch_x[0],batch_x[1]], training = True)
				loss_val = tf.keras.losses.MSE(batch_y, y_pred)

			# calculate the gradients of loss w.r.t. parameters for the batch
			grads = tape.gradient(loss_val, model.trainable_weights)

			# optimize parameters 
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			epoch_loss.append(loss_val)

			if step % 10 == 0:
				print('Step: ', step, ', Loss: ', loss_val.numpy().mean())

			step = step + 1

		print('Average Loss for Epoch: ', epoch, ' : ', np.array(epoch_loss).mean())

		# now get loss for validation set
		n_validation_samples = len(x_validation)
		x_validation_reshaped = np.reshape(x_validation, (2, n_validation_samples, 240, 240, 3)) # (n_samples, 2, 240, 240, 3) -> (2, n_samples, 240, 240, 3)
		validation_preds = model([x_validation_reshaped[0], x_validation_reshaped[1]], training = False)
		validation_loss = tf.keras.losses.MSE(validation_preds, y_validation)
		mean_validation_loss = np.array(validation_loss).mean()
		print('validation loss for Epoch: ', epoch, ' : ', mean_validation_loss)
		
		# save the model if this performs better than the previous model on the validation set
		if mean_validation_loss < prev_validation_loss:
			print('Saving model...')
			model.save('model_skip.h5')
			#np.save("loss_skip.npy", loss, allow_pickle = True, fix_imports = True)
			prev_validation_loss = mean_validation_loss

	return model,loss_list


def getDistance(p1,p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def getNewPoint(point,angle,dist):
	new_x = point[0] + dist*math.cos(angle)
	new_y = point[1] + dist*math.cos(angle)
	return [new_x,new_y]

def test(x_data,y_data,model):

	num_test_samples = len(x_data)
	new_x_data = np.reshape(x_data,(2,num_test_samples,240,240,3))

	preds = model.predict(new_x_data)

	startPointError1 = []
	endPointError1 = []
	startPointError2 = []
	endPointError2 = []

	for i,pred in enumerate(preds):
		predStart = pred[:2]
		trueStart = y_data[i][:2]
		startPointError1.append(getDistance(predStart,trueStart))

		predEnd = getNewPoint(pred[:2],pred[2],pred[3])
		trueEnd = getNewPoint(y_data[i][:2],y_data[i][2],y_data[i][3])
		endPointError1.append(getDistance(predEnd,trueEnd))

		predStart = pred[4:6]
		trueStart = y_data[i][4:6]
		startPointError2.append(getDistance(predStart,trueStart))

		predEnd = getNewPoint(pred[4:6],pred[6],pred[7])
		trueEnd = getNewPoint(y_data[i][4:6],y_data[i][6],y_data[i][7])
		endPointError2.append(getDistance(predEnd,trueEnd))
	
	averageStartPointError1 = sum(startPointError1)/len(startPointError1)
	averageEndPointError1 = sum(endPointError1)/len(endPointError1)
	averageStartPointError2 = sum(startPointError2)/len(startPointError2)
	averageEndPointError2 = sum(endPointError2)/len(endPointError2)

	print("The Average Start Point 1 Error is {} pixels".format(averageStartPointError1))
	print("The Average End Point 1 Error is {} pixels".format(averageEndPointError1))
	print("The Average Start Point 2 Error is {} pixels".format(averageStartPointError2))
	print("The Average End Point 2 Error is {} pixels".format(averageEndPointError2))
	
	return startPointError1,endPointError1,startPointError2,endPointError2

def main():

	epochs = 10
	batch_size = 100
	rope_runs = 68

	print("Getting Data...")
	X,y = get_rope_data(rope_runs)
	np.save("X_skip.npy", X, allow_pickle=True, fix_imports=True)
	np.save("y_skip.npy", y, allow_pickle=True, fix_imports=True)

	# split the data into training, validation and test: (.7, .2, .1) split
	train_index = int(len(X) * 0.7)
	validation_index = int(len(X) * 0.9)

	# training set
	x_train = X[:train_index]
	y_train = y[:train_index]

	# validation set
	x_validation = X[train_index:validation_index]
	y_validation = y[train_index:validation_index]

	# test set
	x_test = X[validation_index:]
	y_test = y[validation_index:]

	#Load Data
	#X = np.load("X.npy")
	#y = np.load("y.npy")

	model, loss = train(x_train, y_train, x_validation, y_validation, epochs, batch_size)

	# model.save("model.h5")
	# np.save("loss.npy", loss, allow_pickle=True, fix_imports=True)


if __name__ == "__main__":
    main()



