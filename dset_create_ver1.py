# imports.
import numpy as np 
from vizdoom import * 
import random 
import time 
from skimage import transform 
from collections import deque 
import matplotlib.pyplot as plt 
import cv2 
from collections import deque

# globals.
frame_x = 240
frame_y = 320
stack_size = 3


# initialize the game.
def init_env():
	game = DoomGame()
	game.load_config('dset_create.cfg')
	game.set_doom_scenario_path('test_map.wad')
	game.init()

	# use in case of fixed buttons.
	backward = [0, 0, 0, 1]
	forward = [0, 0, 1, 0]
	right = [1, 0, 0, 0]
	left = [0, 1, 0, 0]
	all_actions = [right, left, forward, backward]

	# use in case of delta controls
	# right = [100, 0]
	# left = [-100, 0]
	# forward = [0, 100]
	# backward = [0, -100]
	# actions = [right, left, forward, backward]
	
	return game, all_actions

# normalize the frame and reized it.
def norm_resize(frame):
	normalized_frame = frame/255.0
	resized_frame = transform.resize(normalized_frame, [frame_x, frame_y])
	return resized_frame

# doom environment incorporates momentum during movement, there is no discrete action length.
# so, stack frames to give the network a sense of momentum.
def  stack_frames(stacked_frames, frame, is_new_episode):
	# frame = norm_resize(state)

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


input_images_dir = './input_data/'
target_images_dir = './target_data/' 

# main loop.
game, all_actions = init_env()
max_steps = 1000 # actions per episode
episodes = 10

# stacked_state is just the 3 consecutive frames stacked along 2nd dimension. will be fed to the neural network.
# stacked_frame is the 3 consective frames that will be saved for training. 
for episode in range(episodes):
	print('================================')
	print('Episode : ', episode)
	game.new_episode()
	step = 0
	# initialize the starting frames with zeros for the start of every episode
	stacked_frames = deque([np.zeros((frame_x, frame_y), dtype=np.int) for i in range(stack_size)], maxlen=3)
	while  step < (max_steps - 1):
		# new episode
		print('____________________________')
		print('Step : ', step)
		if step == 0: # if this is the first step of the episode, get the starting frame and stack it. 
			# get the state at the start of the step
			print('Initialize episode...')
			state = game.get_state()
			frame = state.screen_buffer # (240, 320, 3)
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # (240, 320, 1)
			stacked_state, stacked_frames = stack_frames(stacked_frames, frame, True) # stacked_frame -> (240, 320, 3)


		if game.is_episode_finished() == False:
			# get the position of the agent before taking the action
			before_x = game.get_game_variable(GameVariable.POSITION_X)
			before_y = game.get_game_variable(GameVariable.POSITION_Y)

			# get the action to append into the training input data
			misc = state.game_variables
			action = random.choice(all_actions)

			# if first step in episode, then, the stacked state created above will be stored as input state for this step. 
			# otherwise, the stacked state after action was performed in previous step will be the new input state. 
			# store the input_image, the filename will be as follows -> ep_<episodenumber>_step_<stepnumber>_<before_x>_<before_y>_<action>_.png
			input_fname = 'ep_' + str(episode) + '_step_' + str(step) + '_' + str(before_x) + '_' + str(before_y) + '_' + str(np.argmax(action)) + '_.png'
			input_save_path = input_images_dir + input_fname
			res = cv2.imwrite(input_save_path, stacked_state)
			print('Saving : ', input_save_path)	
			print('Saved input state.')

			# perform the action
			reward = game.make_action(action)
			print('Performed action.')

			state = game.get_state()
			frame = state.screen_buffer # (240, 320, 3)
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # (240, 320)
			target_frame = np.reshape(frame, (240, 320, 1))

			# get the position of the agent after performing the action
			after_x = game.get_game_variable(GameVariable.POSITION_X)
			after_y = game.get_game_variable(GameVariable.POSITION_Y)

			# save the target state 
			target_fname = 'ep_' + str(episode) + '_step_' + str(step) + '_' + str(after_x) + '_' + str(after_y) + '_.png'
			target_save_path = target_images_dir + target_fname
			res = cv2.imwrite(target_save_path, target_frame)
			print('Saving : ', target_save_path)
			print('Saved target state.')

			# finally append the state reached to our deque to get the next input state
			stacked_state, stacked_frames = stack_frames(stacked_frames, frame, False)
			
		else:
			print('Reached end of episode.')
			break	
		
		time.sleep(1)

		step = step + 1

	time.sleep(2)

game.close()
