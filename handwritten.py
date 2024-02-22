# defining num_classes

num_classes = 10
batch_size = 100

# IMPORT LIBRARIES AND DATASET 

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten 
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K

# split the data of training and testing sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# DATA PREPROCESSING

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

# converting class vectors to matrices of binary class

y_train = keras.utils.to_categorical(y_train, num_classes)              # recheck here for num_classes
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# creating the model

batch_size = 128
num_classes = 10
epochs = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# training the model

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
print("The model has successfully trined")
model.save('mnist.h5')
print("Saving the bot as mnist.h5")

# evaluating the model

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# creating GUI to predict digits

from keras.models import load_model
from Tkinter import *
import Tkinter as tk 
import win32gui
from PIL import ImageGrab, Image 
import numpy as np
model = load_model('mnist.h5')
def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28,28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping for model normalization
    img = img.reshape(1, 28, 28, 1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # creating elements
        self.canvas = tk.Canvas(self, width=200, height=200, bg = "black", cursor="cross")
        self.label = tk.Label(self, text="Analyzing..", font=("Helvetica", 48))
        self.classify_btn =tk.Button(self, text ="Searched", command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Dlt", command = self.clear_all)

        #grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2 )
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2 )
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("", self.start_pos)
        self.canvas.bind("", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        Hd = self.canvas.winfo_id()         # to fetch the handle of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit)+ ',' + str(int(acc*100))+'%')
    def draw_lines(slf, event):
        slf.x = event.x
        slf.y = event.y
        r=8
        slf.canvas.create_oval(slf.x-r, slf.y-r, slf.x+r, slf.y+r, fill='black')
    app = App()
    mainloop()