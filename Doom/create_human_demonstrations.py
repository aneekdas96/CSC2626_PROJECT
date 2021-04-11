import numpy as np 
from vizdoom import * 
import random 
import time 
from skimage import transform 
from collections import deque 
import matplotlib.pyplot as plt 
import cv2 
from collections import deque

# root path for test scenarios
test_path = './test/'
# each folder will contain the target images for all steps in the scenario. also, a file that contains the final position of the
# agent at the end of the scenario.

# num of scenarios to test on, each scenario has num_action steps
num_scenarios = 100
num_actions = 20

# all actions
backward = [0, 0, 0, 1]
forward = [0, 0, 1, 0]
right = [1, 0, 0, 0]
left = [0, 1, 0, 0]
all_actions = [right, left, forward, backward]


# function to initialize the environment before each scenario
def init_env():
	game = DoomGame()
	game.load_config('dset_create.cfg')
	game.set_doom_scenario_path('test_map.wad')
	game.init()
	return game

game = init_env()


# generate n test scenarios, each with 10 actions
def get_test_scenario_actions():
	test_scenario_actions = []
	# generate 100 test scenario actions
	for i in range(num_scenarios):
		actions_in_scenario = []
		for j in range(num_actions):
			action_choice = np.random.choice(len(all_actions))
			action = all_actions[action_choice]
			actions_in_scenario.append(action)
		test_scenario_actions.append(actions_in_scenario)

	test_scenario_actions = np.array(test_scenario_actions).astype(np.float32)
	return test_scenario_actions


# Run a scenario in our doom environment and store all the observations (will be used as human demonstrations)
def get_target_states_for_scenario(scenario_num, test_scenario_actions):
	game.new_episode()
	end_pos_fname = test_path + str(scenario_num) + '/pos.txt' # will hold the final position of the agent for a scenario (target location)
	actions_in_scenario = test_scenario_actions[scenario_num]
	for i in range(num_actions): # represents the step number in the episode
		# perform action for the step
		action = list(actions_in_scenario[i])
		print('The action is : ', action)
		reward = game.make_action(action)
		print('Performed action.')

		state = game.get_state()
		frame = state.screen_buffer # (240, 320, 3)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # (240, 320, 1)
		target_image_single_channel = frame
		target_image = np.append(frame, frame)
		target_image = np.append(target_image, target_image_single_channel)
		target_image = np.reshape(target_image, (240, 320, 3))

		# get the position of the agent in the environment 
		pos_x = game.get_game_variable(GameVariable.POSITION_X)
		pos_y = game.get_game_variable(GameVariable.POSITION_Y)			

		target_fname = test_path +  str(scenario_num) + '/' + str(i) + '_.png'
		res = cv2.imwrite(target_fname, target_image)
		print('Saving : ', target_fname)
		print('Saved target state.')
		time.sleep(0.1)
	# get the final position of the agent in the environment and write it to file
	final_x = game.get_game_variable(GameVariable.POSITION_X)
	final_y = game.get_game_variable(GameVariable.POSITION_Y)				
	f = open(end_pos_fname, 'w')
	f.write(str(final_x) + ',' + str(final_y))
	f.close()


# create the test dataset (iterate through the scenarios and execute the actions that are considered as human demonstrations)
def create_human_demonstrations():
	test_scenario_actions = get_test_scenario_actions()
	for i in range(num_scenarios):
		get_target_states_for_scenario(i, test_scenario_actions)
		time.sleep(1)
	print('Done creating targets to be used as human demonstrations.')
	game.close()

create_human_demonstrations()

