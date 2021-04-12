# CSC2626_PROJECT
CSC2626 Project: Using Self-supervised learning with imitation learning for deformable object manipulation.

## Doom Experiment
Instructions for installing Vizdoom
1. Please visit the following link and follow instructions to install the Vizdoom package for your python version and OS. 
   https://github.com/mwydmuch/ViZDoom

2. Please run create_doom_dataset.py. This will create the dataset for the model to train on. 

3. Please run create_human_demonstrations.py. This will create scenarios with human demonstrations

4. Please run doom_train.py. This will train the model on our training dataset created by running step 2.

5. Please run run_doom_model_on_scenarios.py to evaluate our model on scenarios where intermediate steps are provided as human demonstrations. 

Notes:
1. All hyperparameters used for the experiments are declared after the imports section of each file. Please feel free to change them
   to replicate all results mentioned in our report.

2. Please contact us at aneekdas@cs.toronto.edu and jackellis@cs.toronto.edu if you have any queries regarding the implementation/ execution. 

## Rope Experiment
**model.py:** Model file for training on non-skipped sequence of images  
**model_skip.py:** Model file for training on skipped sequence of images  
**train.py:** Script to train the original rope model. Can change the number of epochs, batch size, and amount of data  
**train_skip.py:** Script to train the skipping rope model. Can change the number of epochs, batch size, and amount of data  

## Ball Experiment
**clean_ball_data.py:** Contains functions to create image sequences from raw images  
**model.py:** Model file for training on sequence of images  
**train_ball.py:** Script to train the ball model. Can change the number of epochs, batch size, and amount of data  
**test_ball.py:** Evaluates the saved model  
