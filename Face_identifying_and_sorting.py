import cv2
import os
import shutil
from cv2 import IMWRITE_PNG_STRATEGY_FIXED
import numpy as np
from random import shuffle
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


training_source = "source"              #訓練數據位置

data_need_identify = "data_need_to_be_sort/"    #需要分析的數據的位置

file_destination = 'result'                     #分析完的數據的放置位置
file_unknown_destination = 'result_unknown'

num_files_source= sum(os.path.isfile(os.path.join(training_source, f)) for f in os.listdir(training_source))         #偵測訓練數據的數量
num_files_identify = sum(os.path.isfile(os.path.join(data_need_identify, f)) for f in os.listdir(data_need_identify))  #偵測需判斷數據的數量



#將訓練資料分出正確&不正確
def my_label(image_name):
    name = image_name.split('.')[-3]       # 以'.'為基準拆分圖片名 => 之後抓取倒數第三位的值 => Hikari_Mori. _02 . jpg  => 抓Hikari_Mori
    
    # if you have two person or target in your dataset
    if name == "Hikari_Mori":
        return np.array([1,0])
    else:
        return np.array([0,1])

# 訓練數據位置
def my_data():
    data = []
    for img in tqdm(os.listdir(training_source)):
        path = os.path.join(training_source, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)
    return data
data = my_data()


# training 神經網路
train = data[int(num_files_source/3):]    #訓練數量 => 目前設定為約2/3數量的檔案拿來train
test = data[:int(num_files_source/3)]     #測試數量 => 1/3數量的檔案拿來test
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
y_test = [i[1] for i in test]

tf.compat.v1.reset_default_graph()
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
# 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)
model.fit(X_train, y_train, n_epoch=5, validation_set=(X_test, y_test), show_metric = True, run_id="FRS" )  #n_epoch => 訓練次數


"""
# Original
# 測試移動指定數量
img_num = 38349
for i in range(num_files_identify+1000):         #驗證數量(+1000是因為有些名子後面會是_1或_2  這些要再特別偵測有點麻煩 就直接多加一些偵測數量 反正空的就會跳過而已)
    if os.path.isfile( data_need_identify + str(img_num) + "_0.jpg") == False:
        img_num += 1
    elif os.path.isfile( data_need_identify + str(img_num) + "_0.jpg") == True:
        Vdata = []
        path = os.path.join( data_need_identify + str(img_num) + "_0.jpg")
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))

        number = str(img_num) + "_0.jpg"
        img_number = number.split('.')[0]

        Vdata.append([np.array(img_data), img_number])
        
        data = img_data.reshape(50,50,1)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            my_label = "Hikari_Mori"
            shutil.move( data_need_identify + str(img_num) + "_0.jpg", file_destination)
        else:
            my_label = 'Unknown'
            shutil.move( data_need_identify + str(img_num) + "_0.jpg", 'test')
        img_num = img_num + 1
"""

# 測試移動指定數量
print("identifying........")
def identify():
    img_num = 0             #起始檔案的編號

    for i in range(num_files_identify):
        if i >= 0 and i < 10:                   #因為如果是第1個檔案(00001_01) 他會抓成1_0.jpg => 所以根據他的數字大小來重新編號 讓編號10000之前的檔案也能被偵測到
            img_num = "0000%d" %i
        elif i >= 10 and i < 100:
            img_num = "000%d" %i
        elif i >= 100 and i < 1000:
            img_num = "00%d" %i
        elif i >= 1000 and i < 10000:
            img_num = "0%d" %i
        else:
            img_num = img_num


        if os.path.isfile( data_need_identify + str(img_num) + "_0.jpg") == True:
            Vdata = []
            path = os.path.join( data_need_identify + str(img_num) + "_0.jpg")
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (50,50))

            number = str(img_num) + "_0.jpg"
            img_number = number.split('.')[0]

            Vdata.append([np.array(img_data), img_number])
            
            data = img_data.reshape(50,50,1)
            model_out = model.predict([data])[0]

            if np.argmax(model_out) == 0:
                my_label = "Hikari_Mori"
                shutil.move( data_need_identify + str(img_num) + "_0.jpg", file_destination)
            else:
                my_label = 'Unknown'
                shutil.move( data_need_identify + str(img_num) + "_0.jpg", file_unknown_destination)
            
            img_num = int(img_num) + 1


        elif os.path.isfile( data_need_identify + str(img_num) + "_1.jpg") == True:
            Vdata = []
            path = os.path.join( data_need_identify + str(img_num) + "_1.jpg")
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (50,50))

            number = str(img_num) + "_1.jpg"
            img_number = number.split('.')[0]

            Vdata.append([np.array(img_data), img_number])
            
            data = img_data.reshape(50,50,1)
            model_out = model.predict([data])[0]

            if np.argmax(model_out) == 0:
                my_label = "Hikari_Mori"
                shutil.move( data_need_identify + str(img_num) + "_1.jpg", file_destination)
            else:
                my_label = 'Unknown'
                shutil.move( data_need_identify + str(img_num) + "_1.jpg", file_unknown_destination)

            img_num = int(img_num) + 1


        elif os.path.isfile( data_need_identify + str(img_num) + "_2.jpg") == True:
            Vdata = []
            path = os.path.join( data_need_identify + str(img_num) + "_2.jpg")
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (50,50))

            number = str(img_num) + "_2.jpg"
            img_number = number.split('.')[0]

            Vdata.append([np.array(img_data), img_number])
            
            data = img_data.reshape(50,50,1)
            model_out = model.predict([data])[0]

            if np.argmax(model_out) == 0:
                my_label = "Hikari_Mori"
                shutil.move( data_need_identify + str(img_num) + "_2.jpg", file_destination)
            else:
                my_label = 'Unknown'
                shutil.move( data_need_identify + str(img_num) + "_2.jpg", file_unknown_destination)

            img_num = int(img_num) + 1
        

        else:
            img_num = int(img_num) + 1
        
        #if (os.path.isfile( data_need_identify + str(img_num) + "_0.jpg")) == False or (os.path.isfile( data_need_identify + str(img_num) + "_1.jpg")) == False or (os.path.isfile( data_need_identify + str(img_num) + "_2.jpg")) == False:
        #img_num = int(img_num) + 1          #因為偵測&重編號後會變成字串 所以把它弄回int之後在+1之後繼續for迴圈

identify()
identify()
identify()   #放三個 identify() 是因為 當編號有重複2個以上的圖片時 若只identify一次則第二個以後的就不會被偵測 ( Ex: 00050_0, 00050_1, 00050_2 )


print("\n")
input("Identifying and sorting complete!!!")