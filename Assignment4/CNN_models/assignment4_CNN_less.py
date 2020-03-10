import numpy as np
import pandas as pd


#load data (change filepaths if necessary)
train_images = np.load('winter2020-mais-202/train_images.npy')
test_images = np.load('winter2020-mais-202/test_images.npy')

train_labels = np.loadtxt('winter2020-mais-202/train_labels.csv', delimiter=',', skiprows=1)[:, 1]

#import modules
from keras.utils import to_categorical #to encode the labels
from keras.models import Sequential #we use the sequential model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D #model layers
from keras.losses import categorical_crossentropy

#reshaping data
x_train = train_images.reshape(50000,28,28,1)
x_test = test_images.reshape(20000,28,28,1)

#we need to encode the labels to train our model
y_train = to_categorical(train_labels)

#Model
model = Sequential()
# Add convolution 2D
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, validation_split=0.25, epochs=7)

y_test = model.predict(x_test)
y_test = np.argmax(y_test ,axis=1) #decoding

df_test = pd.read_csv('winter2020-mais-202/sample_submission.csv')
df_test['label'] = y_test
df_test.to_csv('winter2020-mais-202/submission.csv', index=False)