# CSC2626_PROJECT
CSC2626 Project: Using Self-supervised learning with imitation learning for deformable object manipulation.

We prefer that you use an Anaconda environment for running our project.
Please run ' pip install -r requirements.txt ' inside your environment on your command prompt to install the necessary packages.

## Doom Experiment
Instructions for installing Vizdoom
1. Please visit the following link and follow instructions to install the Vizdoom package for your python version and OS. 
   https://github.com/mwydmuch/ViZDoom

2. Please run create_doom_dataset.py. This will create the dataset for the model to train on. 

3. Please run create_human_demonstrations.py. This will create scenarios with human demonstrations

4. Please run doom_train.py. This will train the model on our training dataset created by running step 2.

5. Please run run_doom_model_on_scenarios.py to evaluate our model on scenarios where intermediate steps are provided as human demonstrations. 

## Rope Experiment
To run the vanilla rope experiment
1. Please run train.py. This will read the dataset, load the model architecture, train the model (defined in model.py) and plot the loss curves.

To run the rope experiment with skipped images
1. Please run train_skip.py. This will read the dataset, load the model architecture, train the model (defined in model_skip.py) and plot the loss curves.

## Ball Experiment
To run the experiment on the ball dataset
1. First, please run clean_ball_data.py. This will create image sequences from raw images.
2. Please run train_ball.py. This will train the model on the dataset and plot loss curves.
3. Please run test_ball.py to evaluate the model.


Notes:
1. All hyperparameters used for the experiments are declared after the imports section of each file. Please feel free to change them
   to replicate all results mentioned in our report.

2. Please contact us at aneekdas@cs.toronto.edu and jackellis@cs.toronto.edu if you have any queries regarding the implementation/execution or need access to any/all of the 3 datasets.
