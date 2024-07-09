import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

dataset=[]
label=[]

image_dir = 'datasets/'

no_tumor= os.listdir(image_dir+'no/')
yes_tumor= os.listdir(image_dir+'yes/')

#print(no_tumor)

#path='no0.jpg'
#print(path.split('.')[1])

for i , image_name in enumerate(no_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)




dataset=np.array(dataset)
label=np.array(label)

x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=0)

print(x_train.shape)
