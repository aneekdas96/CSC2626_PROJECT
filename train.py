import cv2
import numpy as np
import os
from model import siamese_net
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy

def get_data_rope_run(runNumber):

    x_data = []
    y_data = []
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

    return np.array(x_data),np.array(y_data)

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

def create_batch(batch_size,x_train,y_train):

    batch_x = np.zeros((2,batch_size,240,240,3))
    batch_y = np.zeros((batch_size,4))

    num_samples = len(x_train)
    for i in range(batch_size):
        new_sample = np.random.randint(0,num_samples)
        batch_x[0][i] = x_train[new_sample][0]
        batch_x[1][i] = x_train[new_sample][1]
        batch_y[i] = y_train[new_sample]
    
    return batch_x,batch_y



def train(X,y,epochs,batch_size):

    train_split = 0.7
    train_index = int(len(X)*0.7)
    x_train = X[:train_index]
    y_train = y[:train_index]
    x_test = X[train_index:]
    y_test = y[train_index:]

    model = siamese_net()
    optimizer= Adam(learning_rate=0.0006)
    model.compile(loss='mae', optimizer=optimizer)

    print("Training...")
    loss_list = []
    for epoch in range(1,epochs+1):
        batch_x,batch_y = create_batch(batch_size,x_train,y_train)
        loss = model.train_on_batch([batch_x[0],batch_x[1]],batch_y)
        loss_list.append(loss)
        print('Epoch:', epoch, ', Loss:',loss)

    return model,loss_list


def main():

    epochs = 5000
    batch_size = 100
    rope_runs = 10

    print("Getting Data...")
    X,y = get_rope_data(rope_runs)
    np.save("X.npy",X,allow_pickle=True, fix_imports=True)
    np.save("y.npy",y,allow_pickle=True, fix_imports=True)

    #Load Data
    #X = np.load("X.npy")
    #y = np.load("y.npy")

    model,loss = train(X,y,epochs,batch_size)

    model.save("model.h5")
    np.save("loss.npy",loss,allow_pickle=True, fix_imports=True)


if __name__ == "__main__":
    main()



