import numpy as np
import cv2 
import os
from matplotlib import pyplot as plt
import random
import pickle
import math

#input_folder = "Raw_Data"
#output_folder = "intermediate_data"

PIC_COORDS = []

def click_event(event, x, y, flags, params): 
    if event == cv2.EVENT_LBUTTONDOWN: 
        PIC_COORDS.append([x,y])

def resize_ball_images(input_folder,output_folder):

    count = 0
    for file_name in os.listdir(input_folder):
        image_file = os.path.join(input_folder,file_name)
        img = cv2.imread(image_file)
        resized_img = cv2.resize(img, (240,240))
        outfile = "{}_ball.jpg".format(count)
        out_path = os.path.join(output_folder,outfile)
        cv2.imwrite(out_path,resized_img)
        count += 1

def show_images(input_folder):

    for file_name in os.listdir(input_folder):
        image_file = os.path.join(input_folder,file_name)
        img = cv2.imread(image_file)
        cv2.imshow('image',img)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def create_point(point,variance = 4):
    x = point[0]
    y = point[1]

    new_x = np.random.normal(x,variance)
    new_y = np.random.normal(y,variance)

    return [new_x,new_y]

def show_coords(input_folder):
    coords = np.load(os.path.join(input_folder,'coords.npy'))

    count = 0
    for file_name in os.listdir(input_folder):
        image_file = os.path.join(input_folder,file_name)
        img = cv2.imread(image_file)
        print(coords[count])
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        count += 1

def create_numpy_list(input_folder):
    coords = np.load(os.path.join(input_folder,'coords.npy'))
    master_list = {}

    count = 0
    for file_name in os.listdir(input_folder):
        if file_name != 'coords.npy':
            master_list[file_name] = coords[count]
            count += 1
    outfilename = os.path.join(input_folder,"file_name_coords.pickle")
    outfile = open(outfilename,'wb')
    pickle.dump(master_list,outfile)
    outfile.close()

def show_master_list(input_folder):
    infile = open(os.path.join(input_folder,"file_name_coords.pickle"),'rb')
    new_dict = pickle.load(infile)
    infile.close()
    print(new_dict)

def get_line_length(x1,x2,y1,y2):
    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)*0.38/240
    return length

def get_angle(x1,x2,y1,y2):

    v1 = [-1,0]
    v2 = [x2-x1,y2-y1]

    uv1 = v1/np.linalg.norm(v1)
    uv2 = v2/np.linalg.norm(v2)
    dot_product = np.dot(uv1,uv2)
    angle = np.arccos(dot_product)
    
    if v2[1]>=0:
        return angle
    else:
        return 2*np.pi - angle

def create_actions(coords):
    actions = []
    N = len(coords)

    for i in range(1,N):
        start_x = coords[i-1][0]
        start_y = coords[i-1][1]
        end_x = coords[i][0]
        end_y = coords[i][1]

        length = get_line_length(end_x,start_x,end_y,start_y)
        angle = get_angle(start_x,end_x,-start_y,-end_y)

        actions.append([start_x,start_y,angle,length,1])
    
    return np.array(actions)

def create_run(runNumber,N):
    newDir = os.path.join("ball","run{}".format(runNumber))
    if "run{}".format(runNumber) not in os.listdir("ball"):
        os.mkdir(newDir)
    int_folder = "intermediate_data"

    infile = open(os.path.join('intermediate_data',"file_name_coords.pickle"),'rb')
    file_coords_dict = pickle.load(infile)
    infile.close()

    coord_list = []
    for i in range(N):
        random_file = random.choice(list(file_coords_dict.keys()))
        newCoords = create_point(file_coords_dict[random_file])

        image_file = os.path.join(int_folder,random_file)
        img = cv2.imread(image_file)
        num = str(i).zfill(4)
        outpath = os.path.join(newDir,"img_{}.jpg".format(num))
        cv2.imwrite(outpath,img)
        coord_list.append(newCoords)

    actions = create_actions(coord_list)

    action_file = os.path.join(newDir,"actions.npy")
    np.save(action_file,np.array(actions))
    

def main():
    #raw_folder = "Raw_Data"
    int_folder = "intermediate_data"                #Whatever the name of the ball function is
    #resize_ball_images(raw_folder,int_folder)
    #show_images(int_folder)
    #ball_locations = np.array(PIC_COORDS)
    #coord_path = os.path.join(int_folder,'coords.npy')
    #np.save(coord_path,ball_locations)
    #show_coords(int_folder)
    #create_numpy_list(int_folder)
    #show_mater_list(int_folder)
    create_run(0,N=1000)                    #Use this function to create ball runs



if __name__ == '__main__':
    main()