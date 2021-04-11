import numpy as np 
from vizdoom import * 
import random 
import time 
from skimage import transform 
from collections import deque 
import matplotlib.pyplot as plt 
import cv2 
from collections import deque
from model_doom import dual_net
import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from math import sqrt

# dimension of frames
frame_x = 240
frame_y = 320
stack_size = 3 # both left input (input state) and right input (target state) have 3 channels

# root path for test scenarios
test_path = './test/'

# store all the rms_distance values between target and reached states for all scenarios
mean_distances = []

# num of scenarios to test on, each scenario has num_action steps
num_scenarios = 100
num_steps = 20

# all actions
backward = [0, 0, 0, 1]
forward = [0, 0, 1, 0]
right = [1, 0, 0, 0]
left = [0, 1, 0, 0]
all_actions = [right, left, forward, backward]

# load the saved model
model = tf.keras.models.load_model('model_doom.h5')

# function to initialize the environment before each scenario
def init_env():
	game = DoomGame()
	game.load_config('dset_create.cfg')
	game.set_doom_scenario_path('test_map.wad')
	game.init()
	return game

game = init_env()


# function to stack frames 
def  stack_frames(stacked_frames, frame, is_new_episode):
	if is_new_episode:
		# initialize our deque with all zeros for a new episode
		stacked_frames = deque([np.zeros((frame_x, frame_y), dtype=np.int) for i in range(stack_size)], maxlen=3)
		# stack the same from for stack-length for first step
		for idx in range(stack_size):
			stacked_frames.append(frame)
	else:
		# stack the new frame to our state and remove the oldest state from the deque
		stacked_frames.append(frame)
	# get the state
	stacked_state = np.stack(stacked_frames, axis=2)
	return stacked_state, stacked_frames


# test on the scenarios with human demonstrations
def test_scenarios():
	for scenario_num in range(num_scenarios):
		print('In scenario : ', scenario_num)
		game.new_episode()
		stacked_frames = deque([np.zeros((frame_x, frame_y), dtype=np.int) for i in range(stack_size)], maxlen=3)
		for i in range(num_steps):
			print('Step : ', i)
			if i==0: # first state of the episode, then stack the same frame thrice for our input state
				state = game.get_state()
				frame = state.screen_buffer # (240, 320, 3)
				frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # (240, 320, 1)
				stacked_state, stacked_frames = stack_frames(stacked_frames, frame, True)

			if game.is_episode_finished() == False:
				# load our human demonstration, will be passed as input through the right input of our model
				target_fname = test_path +  str(scenario_num) + '/' + str(i) + '_.png'
				target_state = cv2.imread(target_fname)

				# reshape the data to include batch dimension (make it compatible with models input layer)
				left_input = np.reshape(stacked_state, (1, frame_x, frame_y, stack_size))
				right_input = np.reshape(target_state, (1, frame_x, frame_y, stack_size))
				
				# predict the action to go from input_state to target_state
				action_preds = model.predict([left_input, right_input])
				action_choice = np.argmax(action_preds)
				action = np.zeros(4)
				action[action_choice] = 1
				action = list(action.astype(float))
				print('The predicted action for step i : ', action)

				# perform the predicted action to get the next frame. This will be our next input frame.
				reward = game.make_action(action)

				# get our input frame, will be passed as input through the left input of our model after stacking with previous frames
				state = game.get_state()
				frame = state.screen_buffer # (240, 320, 3)
				frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # (240, 320, 1)

				# finally append the state reached to our deque to get the next input state
				stacked_state, stacked_frames = stack_frames(stacked_frames, frame, False)
			else:
				print('Reached end of episode.')
				break

			# get the target position (position reached by performing human demonstrations)
			end_pos_fname = test_path + str(scenario_num) + '/pos.txt'
			f = open(end_pos_fname, 'r')
			store = f.readline()
			store = store.split(',')
			target_x, target_y = float(store[0]), float(store[1])
			f.close()

			# now get the final position reached by the agent
			reached_x = game.get_game_variable(GameVariable.POSITION_X)
			reached_y = game.get_game_variable(GameVariable.POSITION_Y)
			
			# calculate distance between the reached and target states
			rms_distance = sqrt((target_x-reached_x)**2 + (target_y-reached_y)**2)
			mean_distances.append(rms_distance)
			time.sleep(1)
	game.close()

test_scenarios()
print('The mean distance between target and reached states is : ', np.array(mean_distances).mean())