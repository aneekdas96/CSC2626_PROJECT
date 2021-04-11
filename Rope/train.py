import cv2
import numpy as np
import os
from model import dual_net
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 

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
    
    for i in range(len(img_list)-1):
        x_data.append([img_list[i],img_list[i+1]])
        y_data.append(actions[i][:4]) #Dont take last index

    return np.array(x_data), np.array(y_data)

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
            x_data = np.concatenate([x_data,x_temp])
            y_data = np.concatenate([y_data,y_temp])
        print("Retrived Run{}".format(runNum))
    
    #x_data = np.array(x_data)
    #y_data = np.array(y_data)

    rng_state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(rng_state)
    np.random.shuffle(y_data)
    
    return x_data,y_data

def create_batch(batch_size, step, x, y):

    batch_x = np.zeros((2, batch_size, 240, 240, 3))
    batch_y = np.zeros((batch_size, 4))

    batch_start_index = step * batch_size
    batch_stop_index = (step + 1) * batch_size - 1

    # get the samples in the batch
    counter = 0
    while batch_start_index < batch_stop_index:
    	batch_x[0][counter] = x[batch_start_index][0]
    	batch_x[1][counter] = x[batch_start_index][1]
    	batch_y[counter] = y[batch_start_index]
    	counter = counter + 1
    	batch_start_index = batch_start_index + 1

    return batch_x, batch_y



def train(x_train, y_train, x_validation, y_validation, epochs, batch_size):

	prev_validation_loss = 1000000 # max loss (used to save model with lowest validation loss)

	# initialize model and optimizer
	model = dual_net()
	optimizer= Adam(learning_rate=0.0001)

	# number of steps per epoch
	max_steps = int(len(x_train)/batch_size)

	print('Number of steps per epoch : ', max_steps)

	print("Training...")
	total_training_loss = []
	total_validation_loss = []

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
			total_step_loss = loss_val.numpy().mean()
			epoch_loss.append(total_step_loss)

			if step % 10 == 0:
				print('Step: ', step, ', Loss: ', loss_val.numpy().mean())

			step = step + 1

		print('Average Loss for Epoch: ', epoch, ' : ', np.array(epoch_loss).mean())

		# append epoch loss to the list of training losses
		total_training_loss.append(np.array(epoch_loss).mean())

		# now get loss for validation set
		n_validation_samples = len(x_validation)
		print('The number of validation samples : ', n_validation_samples)
		
		max_validation_steps = int(len(x_validation)/batch_size) ###################
		print('The number of steps in total : ', max_validation_steps)

		# x_validation_reshaped = np.reshape(x_validation, (2, n_validation_samples, 240, 240, 3)) # (n_samples, 2, 240, 240, 3) -> (2, n_samples, 240, 240, 3)
		validation_losses = []
		validation_step = 0
		while validation_step < max_validation_steps:
			print('Evaluating on validation set...')
			validation_batch_x, validation_batch_y = create_batch(batch_size, validation_step, x_validation, y_validation)
			# print('Inside validation batch : ', validation_step)
			# print('Shape of validation x : ', validation_batch_x.shape)
			# print('Shape of validation y : ', validation_batch_y.shape)
			validation_y_pred = model([validation_batch_x[0], validation_batch_x[1]], training = False)
			validation_step_loss = tf.keras.losses.MSE(validation_y_pred, validation_batch_y)
			validation_losses.append(np.array(validation_step_loss).mean())
			validation_step = validation_step + 1

		mean_validation_loss = np.array(validation_losses).mean()
		print('Mean Validation Loss for Epoch : ', epoch, ' : ', mean_validation_loss)
		total_validation_loss.append(mean_validation_loss)


		# validation_preds = model([x_validation_reshaped[0], x_validation_reshaped[1]], training = False)
		# validation_loss = tf.keras.losses.MSE(validation_preds, y_validation)
		# mean_validation_loss = np.array(validation_loss).mean()
		# print('validation loss for Epoch: ', epoch, ' : ', mean_validation_loss)
		
		# save the model if this performs better than the previous model on the validation set
		if mean_validation_loss < prev_validation_loss:
			print('Saving model...')
			model.save('model.h5')
			# np.save("loss.npy", loss, allow_pickle = True, fix_imports = True)
			prev_validation_loss = mean_validation_loss

	return model, total_training_loss, total_validation_loss


def main():

	epochs = 10
	batch_size = 100
	rope_runs = 68

	print("Getting Data...")
	X,y = get_rope_data(rope_runs)
	np.save("X.npy", X, allow_pickle=True, fix_imports=True)
	np.save("y.npy", y, allow_pickle=True, fix_imports=True)

	#Load Data
	print('Loading Data...')
	#X = np.load("X.npy")
	#y = np.load("y.npy")

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

	model, total_training_loss, total_validation_loss = train(x_train, y_train, x_validation, y_validation, epochs, batch_size)

	print('The losses for training set : ', total_training_loss)

	# plot the training losses
	loss_df = pd.DataFrame()
	loss_df['epochs'] = list(range(epochs))
	loss_df['validation_loss'] = total_validation_loss
	loss_df['training_loss'] = total_training_loss
	loss_df = loss_df.set_index('epochs')
	sns.lineplot(data = loss_df)
	plt.show()

if __name__ == "__main__":
    main()



