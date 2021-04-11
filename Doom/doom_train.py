import cv2
import numpy as np
import os
from model_doom import dual_net
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from os import listdir
from os.path import isfile, join
import time
import pickle as pkl
from tqdm import tqdm

def get_episode_step_action(image_fname):
	# get episode, step and action from fname
	vals = image_fname.split('_')
	episode_num = vals[1]
	step_num = vals[3]
	action = int(vals[6])

	# construct the fname of the target image file
	target_fname = 'ep_' + str(episode_num) + '_step_' + str(step_num) + '_.png'

	return episode_num, step_num, action, target_fname

def load_doom_dataset():
	
	# variables to store the input images, target images and actions 
	input_images = []
	target_images = []
	actions = []

	input_images_dir = './input_data/'
	target_images_dir = './target_data/' 
	input_image_fnames = listdir(input_images_dir)

	num_samples = len(input_image_fnames)
	# num_samples = 5000
	print('Loading dataset...')
	for file_counter in tqdm(range(num_samples)): 
		input_image_fname = input_image_fnames[file_counter]
		# read the image
		input_image_path = input_images_dir + input_image_fname
		input_image = cv2.imread(input_image_path)

		# # show the loaded sample
		# print('Showing the sample...')
		# cv2.imshow('Sample', input_image)
		# cv2.waitKey(0) 

		episode_num, step_num, action, target_image_fname = get_episode_step_action(input_image_fname)
		
		# get the corresponding target image
		target_image_path = target_images_dir + target_image_fname
		target_image = np.reshape(cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE), (240, 320, 1))
		target_image_single_channel = target_image
		target_image = np.append(target_image, target_image)
		target_image = np.append(target_image, target_image_single_channel)
		target_image = np.reshape(target_image, (240, 320, 3))

		# append the input images, target images and actions to their respective arrays
		input_images.append(input_image)
		target_images.append(target_image)
		actions.append(action)

	input_images = np.array(input_images)
	target_images = np.array(target_images)
	actions = np.array(actions)

	input_images = np.reshape(input_images, (num_samples, 240, 320, 3))
	target_images = np.reshape(target_images, (num_samples, 240, 320, 3))
	# one hot encode the actions 
	actions = actions.astype(np.int32)
	n_values = np.max(actions) + 1
	actions = np.eye(n_values)[actions]
	# actions = np.reshape(actions, (num_samples, 1))

	# shuffle the dataset
	print('Shuffling data...')
	rng_state = np.random.get_state()
	np.random.shuffle(input_images)
	np.random.set_state(rng_state)
	np.random.shuffle(target_images)
	np.random.set_state(rng_state)
	np.random.shuffle(actions)

	np.save('input_images.npy', input_images) # (num_samples, 240, 320, 3)
	np.save('target_images.npy', target_images) # (num_samples, 240, 320, 1)
	np.save('actions.npy', actions) # (num_samples, 4)

	print('Finished loading dataset.')
	return input_images, target_images, actions


def create_batch(batch_size, step, input_images, target_images, actions):

    batch_input_images = np.zeros((batch_size, 240, 320, 3))
    batch_target_images = np.zeros((batch_size, 240, 320, 3))
    batch_actions = np.zeros((batch_size, 4))

    batch_start_index = step * batch_size
    batch_stop_index = (step + 1) * batch_size - 1

    # get the samples in the batch
    counter = 0
    while batch_start_index < batch_stop_index:
    	batch_input_images[counter] = input_images[batch_start_index]
    	batch_target_images[counter] = target_images[batch_start_index]
    	batch_actions[counter] = actions[batch_start_index]
    	counter = counter + 1
    	batch_start_index = batch_start_index + 1

    return batch_input_images, batch_target_images, batch_actions


def train(input_images_train, target_images_train, actions_train, input_images_validation, target_images_validation, actions_validation, epochs, batch_size):

	prev_validation_loss = 1000000 # max loss (used to save model with lowest validation loss)

	# initialize model and optimizer
	model = dual_net()
	optimizer= Adam(learning_rate=0.001)

	# number of steps per epoch
	max_steps = int(len(input_images_train)/batch_size)

	print('Number of steps per epoch : ', max_steps)

	print("Training...")
	total_training_loss = []
	total_validation_loss = []

	# training loop
	for epoch in range(1, epochs+1):

		epoch_loss = []
		print('____________________________________________________')
		print('Epoch: ', epoch)
		step = 0

		# loop through all batches. 1 batch/ step. each step updates the parameters after batch_size samples
		while step < max_steps:
			# get batch
			batch_input_images, batch_target_images, batch_actions = create_batch(batch_size, step, input_images_train, target_images_train, actions_train) 

			with tf.GradientTape() as tape:
				actions_pred = model([batch_input_images, batch_target_images], training = True)
				loss_val = tf.keras.losses.categorical_crossentropy(batch_actions, actions_pred)

			# calculate the gradients of loss w.r.t. parameters for the batch
			grads = tape.gradient(loss_val, model.trainable_weights)

			# optimize parameters 
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			total_step_loss = loss_val.numpy().mean()
			epoch_loss.append(total_step_loss)

			if step % 10 == 0:
				print('Step: ', step, ', Loss: ', loss_val.numpy().mean())

			step = step + 1

		print('Mean Training Loss for Epoch: ', epoch, ' : ', np.array(epoch_loss).mean())

		# append epoch loss to the list of training losses
		total_training_loss.append(np.array(epoch_loss).mean())

		# now get loss for validation set
		n_validation_samples = len(input_images_validation)
		print('The number of validation samples : ', n_validation_samples)
		
		max_validation_steps = int(len(input_images_validation)/batch_size) ###################
		print('The number of steps in total : ', max_validation_steps)

		# x_validation_reshaped = np.reshape(x_validation, (2, n_validation_samples, 240, 240, 3)) # (n_samples, 2, 240, 240, 3) -> (2, n_samples, 240, 240, 3)
		validation_losses = []
		validation_step = 0
		while validation_step < max_validation_steps:
			batch_input_images, batch_target_images, batch_actions = create_batch(batch_size, validation_step, input_images_validation, target_images_validation, actions_validation) 
			validation_y_pred = model([batch_input_images, batch_target_images], training = False)
			validation_step_loss = tf.keras.losses.categorical_crossentropy(batch_actions, validation_y_pred)
			validation_losses.append(np.array(validation_step_loss).mean())
			validation_step = validation_step + 1

		mean_validation_loss = np.array(validation_losses).mean()
		print('Mean Validation Loss for Epoch : ', epoch, ' : ', mean_validation_loss)
		total_validation_loss.append(mean_validation_loss)

		# save the model if this performs better than the previous model on the validation set
		if mean_validation_loss < prev_validation_loss:
			print('Saving model...')
			model.save('model_doom.h5')
			# np.save("loss.npy", loss, allow_pickle = True, fix_imports = True)
			prev_validation_loss = mean_validation_loss

	return model, total_training_loss, total_validation_loss


def main():
	
	# hyperparams
	epochs = 10
	batch_size = 100

	# load dataset
	input_images, target_images, actions = load_doom_dataset()

	# split the data into training, validation and test: (.7, .2, .1) split
	train_index = int(len(input_images) * 0.7)
	validation_index = int(len(input_images) * 0.9)

	# training set
	input_images_train = input_images[:train_index]
	target_images_train = target_images[:train_index]
	actions_train = actions[:train_index]
	print('Shape of input training images : ', input_images_train.shape)
	print('Shape of target training images : ', target_images_train.shape)
	print('Shape of actions train : ', actions_train.shape)

	# validation set
	input_images_validation = input_images[train_index:validation_index]
	target_images_validation = target_images[train_index:validation_index]
	actions_validation = actions[train_index:validation_index]
	print('Shape of input validation images : ', input_images_validation.shape)
	print('Shape of target validation images : ', target_images_validation.shape)
	print('Shape of actions validation : ', actions_validation.shape)


	# test set
	input_images_test = input_images[validation_index:]
	target_images_test = target_images[validation_index:]
	actions_test = actions[validation_index:]
	print('Shape of input test images : ', input_images_test.shape)
	print('Shape of target test images : ', target_images_test.shape)
	print('Shape of actions test : ', actions_test.shape)

	model, total_training_loss, total_validation_loss = train(input_images_train, target_images_train, actions_train, input_images_validation, target_images_validation, actions_validation, epochs, batch_size)

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



