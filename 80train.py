
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


os.chdir('Desktop/test/beta1')


# In[3]:


pwd


# In[4]:


Birthday_files=os.listdir("Accept/")
print(Birthday_files)


# In[5]:


from keras.layers import Flatten
from keras.applications.inception_v3 import InceptionV3

model_sq= InceptionV3(weights='imagenet', include_top=False,input_shape=(299,299,3))

model_sq.summary()


# In[1]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[7]:


from keras.models import Model
from keras.layers import GlobalMaxPooling2D, Dense

t=GlobalMaxPooling2D()(model_sq.output)
o=Dense(26,activation = 'softmax')(t)

#Input is same as Model_sq input and output is o
model_int = Model([model_sq.input],o)

from keras.models import Model
model_int.summary()


# In[8]:


#Compiling inception
model_int.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[9]:


from keras.preprocessing.image import ImageDataGenerator


batch_size = 16
val_frac =0.1
train_datagen = ImageDataGenerator(validation_split=val_frac)

train_generator = train_datagen.flow_from_directory(
        r'C:\Users\sam pc\Desktop\test\betaframe1',  # this is the target directory
        target_size=(299, 299),  # all images will be resized to 299x299
        batch_size=batch_size,
        class_mode='categorical', subset="training")
val_generator = train_datagen.flow_from_directory(
        r'C:\Users\sam pc\Desktop\test\betaframe1',  # this is the target directory
        target_size=(299, 299),  # all images will be resized to 299x299
        batch_size=batch_size,
        class_mode='categorical',subset="validation")



# In[12]:


#csvlogger
import keras
from keras.callbacks import CSVLogger
import time

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

csv_logger = CSVLogger('training1.log')
time_callback = TimeHistory()
model_int.fit_generator(train_generator,epochs =5,steps_per_epoch=250,callbacks=[csv_logger,time_callback],validation_data=val_generator, validation_steps=250)
times = time_callback.times


# In[13]:


model_int1=model_int#intact for softmax approach
model_int1.layers.pop()
fin_o  = model_int1.layers[-1].output
model_fin = Model(input=model_int1.input, output=[fin_o])
model_fin.save('modelinception7.h5')


# In[14]:


pwd


# In[15]:


categories = os.listdir(r'C:\Users\sam pc\Desktop\test\beta1')


# In[16]:


categories


# In[17]:


categories1 = categories[0:26]


# In[18]:


categories1


# In[19]:


import numpy as np
import os,shutil
import subprocess
import scipy
import cv2
import matplotlib.pyplot as plt
subprocess.call('ren *.txt *.bat', shell=True)
final_data=np.zeros(shape=(26,40,40,2048))


for idx,c in enumerate(categories1):
    video_files=os.listdir(c+"/")
    for ind,vid in enumerate(video_files):

        arr=np.zeros(shape=(40,299,299,3))
        vid_file=c+"//"+vid

        os.mkdir('frames')
        get_ipython().system('ffmpeg  -i $vid_file -r 10 frames/wk%02d.jpg')

        for i,img in enumerate(os.listdir("frames/")):
            if i <22:
                print(img)
                img=cv2.imread("frames/"+img)
                img=cv2.resize(img,(299,299))
                arr[i]=img
                #print(str(i)+"th image")
                
        shutil.rmtree('frames')
        pred=model_fin.predict(arr)
        del arr
        final_data[idx,ind]=pred
    #print(str(ind)+"th video done")


# In[21]:


final_data.shape


# In[22]:


L1 = categories1
L2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
d = dict(zip(L1,L2))
d


# In[29]:


from keras.utils import to_categorical
cat2index=d
labels=[0 for i in range(40)]+[1 for i in range(40)]+[2 for i in range(40)]+[3 for i in range(40)]+[4 for i in range(40)]+[5 for i in range(40)]+[6 for i in range(40)]+[7 for i in range(40)]+[8 for i in range(40)]+[9 for i in range(40)]+[10 for i in range(40)]+[11 for i in range(40)]+[12 for i in range(40)]+[13 for i in range(40)]+[14 for i in range(40)]+[15 for i in range(40)]+[16 for i in range(40)]+[17 for i in range(40)]+[18 for i in range(40)]+[19 for i in range(40)]+[20 for i in range(40)]+[21 for i in range(40)]+[22 for i in range(40)]+[23 for i in range(40)]+[24 for i in range(40)]+[25 for i in range(40)]

label_hot= to_categorical(labels)


# In[30]:


finale_data = final_data.reshape(-1,40,2048)


# In[31]:


finale_data.shape


# In[32]:


label_hot.shape


# In[33]:


from sklearn.utils import shuffle
X_train_rnn, y_trainHot_rnn = shuffle(finale_data, label_hot, random_state=13)
from sklearn.model_selection import train_test_split
X_train_rnn, X_validation_rnn, y_train_rnn, y_validation_rnn= train_test_split(X_train_rnn, y_trainHot_rnn, test_size=0.2) 


# In[34]:


from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential
#this is the main trainable model , we are going to use ,
model = Sequential()
model.add(LSTM(1024, return_sequences=False,    #spit out a 1024 dimension vector based on given input sequence,
                                              #input sequence is a 40 frame video 
                       input_shape=(40,2048),
                       dropout=0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(26, activation='softmax'))


# In[35]:



model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])


# In[37]:


model.fit(X_train_rnn,y_train_rnn,batch_size=8,epochs=50,validation_data = (X_validation_rnn,y_validation_rnn))


# In[ ]:


##################################Evaluation with the test set


# In[38]:


pwd


# In[39]:


import os
os.chdir('C:\\Users\\sam pc\\Desktop\\test\\alpha1')


# In[40]:


pwd


# In[41]:


categories_t = os.listdir(r'C:\Users\sam pc\Desktop\test\alpha1')


# In[43]:


len(categories_t) 


# In[46]:


import numpy as np
import os,shutil
import subprocess
import scipy
import cv2
import matplotlib.pyplot as plt
subprocess.call('ren *.txt *.bat', shell=True)
final_data=np.zeros(shape=(26,10,40,2048))


for idx,c in enumerate(categories_t):
    video_files=os.listdir(c+"/")
    for ind,vid in enumerate(video_files):

        arr=np.zeros(shape=(40,299,299,3))
        vid_file=c+"//"+vid

        os.mkdir('frames')
        get_ipython().system('ffmpeg  -i $vid_file -r 10 frames/wk%02d.jpg')

        for i,img in enumerate(os.listdir("frames/")):
            if i <22:
                print(img)
                img=cv2.imread("frames/"+img)
                img=cv2.resize(img,(299,299))
                arr[i]=img
                #print(str(i)+"th image")
                
        shutil.rmtree('frames')
        pred=model_fin.predict(arr)
        del arr
        final_data[idx,ind]=pred
    #print(str(ind)+"th video done")


# In[47]:


final_data.shape


# In[48]:


finale_data1 = final_data.reshape(-1,40,2048)


# In[50]:


finale_data1.shape


# In[52]:


res=model.predict(finale_data1)


# In[53]:


res2 = np.argmax(res,axis=1)


# In[54]:


res2


# In[55]:


L1 = categories_t
L2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
d = dict(zip(L1,L2))
d


# In[56]:


from keras.utils import to_categorical
cat2index=d
labels=[0 for i in range(10)]+[1 for i in range(10)]+[2 for i in range(10)]+[3 for i in range(10)]+[4 for i in range(10)]+[5 for i in range(10)]+[6 for i in range(10)]+[7 for i in range(10)]+[8 for i in range(10)]+[9 for i in range(10)]+[10 for i in range(10)]+[11 for i in range(10)]+[12 for i in range(10)]+[13 for i in range(10)]+[14 for i in range(10)]+[15 for i in range(10)]+[16 for i in range(10)]+[17 for i in range(10)]+[18 for i in range(10)]+[19 for i in range(10)]+[20 for i in range(10)]+[21 for i in range(10)]+[22 for i in range(10)]+[23 for i in range(10)]+[24 for i in range(10)]+[25 for i in range(10)]


# In[57]:


labels


# In[62]:


type(labels1)


# In[60]:


type(res2)


# In[61]:


import numpy as np
labels1 = np.asarray(labels)


# In[63]:


labels1


# In[65]:


accuracy = (np.abs(labels1 - res2) < tolerance ).all(axis=(0,2)).mean()


# In[66]:


import sklearn
acc = sklearn.metrics.accuracy_score(labels1, res2)


# In[67]:


acc


# In[ ]:


######################################################
######################################################
##############################softmax method


# In[73]:


model_int1.summary()


# In[75]:


start = time.clock() 
model.predict(finale_data1)
end = time.clock()

print("Time per video: {} ".format((end-start)/len(finale_data1))) 


# In[74]:


len(finale_data1)


# In[76]:


0.004019854148018134 *260

