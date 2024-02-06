import cv2

# Generate dataset with webcam

def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    
    cap = cv2.VideoCapture(1)
    img_id = 0          #起始名稱ID
    
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


            #file_name_path = "data/"+"Ishwar."+str(img_id)+".jpg"

            file_name_path = "data_need_to_be_sort/"+str(img_id) + "_0" +'.jpg'     #原本的
            #file_name_path = "D:/deep/DeepFaceLab_DirectX12/workspace/data_src/aligned/"+str(img_id) + "_0" +'.jpg'

            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
            
            cv2.imshow("Cropped_Face", face)
            if cv2.waitKey(1)==13 or int(img_id)==100:         #掃描圖片數量
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed !!!")


# Create label
import numpy as np

def my_label(image_name):
    name = image_name.split('.')[-3]       # 以'.'為基準拆分圖片名 => 之後抓取倒數第三位的值 => Hikari_Mori. _02 . jpg  => 抓Hikari_Mori
    
    # if you have two person in your dataset
    if name == "Hikari_Mori":
        return np.array([1,0])
    else:
        return np.array([0,1])
    
    
    """
    # if you have three person in your dataset
    if name=="Ishwar":
        return np.array([1,0,0])
    elif name=="Manish":
        return np.array([0,1,0])
    elif name=="Bijay":
        return np.array([0,0,1])
    """

# Create data
import os
from random import shuffle
from tqdm import tqdm

def my_data():
    data = []
    for img in tqdm(os.listdir("source")):
        path = os.path.join("source", img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)
    return data

data = my_data()

train = data[:4000]
test = data[4000:]
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
print(X_train.shape)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
print(X_test.shape)
y_test = [i[1] for i in test]


#Creating the model
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

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
model.fit(X_train, y_train, n_epoch=10, validation_set=(X_test, y_test), show_metric = True, run_id="FRS" )  #n_epoch => 訓練次數




#Visualize the data and make prediction

def data_for_visualization():
    Vdata = []
    """
    for img in tqdm(os.listdir("D:/deep/DeepFaceLab_DirectX12/workspace/data_src/aligned")):
        path = os.path.join("D:/deep/DeepFaceLab_DirectX12/workspace/data_src/aligned", img) 
    """  
    for img in tqdm(os.listdir("data_need_to_be_sort")):
        path = os.path.join("data_need_to_be_sort", img)  

        img_num = img.split('.')[0]     # => 38349_0.jpg => 抓38349_0 
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))
        Vdata.append([np.array(img_data), img_num])
    #shuffle(Vdata)
    return Vdata

Vdata = data_for_visualization()


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,20))
for num, data in enumerate(Vdata[:100]):        #圖片順序
    img_data = data[0]
    y = fig.add_subplot(10,10, num+1)             # 5*5 = 25 => 只能掃25張圖片
    image = img_data
    data = img_data.reshape(50,50,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0:
        my_label = "Hikari_Mori"
    else:
        my_label = 'Unknown'
        
    y.imshow(image, cmap='gray')
    plt.title(my_label)
    
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()