import cv2
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


image_directory = 'datasets/'

no_tumor_images = os.listdir(os.path.join(image_directory, 'no/'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes/'))

datasets = []
labels = []
Input_Size = 64

for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(os.path.join(image_directory, 'no', image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((Input_Size, Input_Size))
        datasets.append(np.array(image))
        labels.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(os.path.join(image_directory, 'yes', image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((Input_Size, Input_Size))
        datasets.append(np.array(image))
        labels.append(1)

datasets = np.array(datasets)
labels = np.array(labels)

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, random_state=0)

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0   #we can even use normalize
x_test = x_test / 255.0


#for categorical -
y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)

# Model Building
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(Input_Size, Input_Size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Model Compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Training
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=True)

# Save the trained model
model.save('BrainTumor10Epochs_Categorical.h5')
