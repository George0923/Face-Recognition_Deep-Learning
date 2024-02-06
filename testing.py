import cv2
import os
import shutil
from cv2 import IMWRITE_PNG_STRATEGY_FIXED
import numpy as np
from random import shuffle
from tqdm import tqdm

num_files= sum(os.path.isfile(os.path.join("source", f)) for f in os.listdir("source"))
print(num_files)


#移動單個檔案
"""
num = 38349
file_source = 'test/' + str(num) + '_0.jpg'
file_destination = 'result'
shutil.move(file_source, file_destination)
"""

#一次移動多個檔案
"""
file_source = 'test'
file_destination = 'result'
files = os.listdir(file_source)
for file in files:
    new_path = shutil.move(f"{file_source}/{file}", file_destination)
    print(new_path)
"""


#測試移動單個的
"""
img_num = 38349
Vdata = []
path = os.path.join("data_need_to_be_sort/" + str(img_num) + "_0.jpg")
img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_data = cv2.resize(img_data, (50,50))

number = str(img_num) + "_0.jpg"
img_number = number.split('.')[0]

Vdata.append([np.array(img_data), img_number])

    #fig = plt.figure(figsize = (20,20))

    #for num, data in enumerate(Vdata):
        #img_data = data[0]
        #y = fig.add_subplot(10,10, num+1)             # 5*5 = 25 => 只能掃25張圖片
        # #image = img_data
    
data = img_data.reshape(50,50,1)
model_out = model.predict([data])[0]

            
if np.argmax(model_out) == 0:
    my_label = "Hikari_Mori"
    shutil.move("data_need_to_be_sort/" + str(img_num) + "_0.jpg", 'result')
else:
    my_label = 'Unknown'
    shutil.move("data_need_to_be_sort/" + str(img_num) + "_0.jpg", 'source')
#img_num = img_num + 1
"""

