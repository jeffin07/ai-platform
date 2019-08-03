from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
import keras.datasets.fashion_mnist as fashion_mnist
import keras
import numpy as np
import mlflow.keras
import argparse

mlflow.keras.autolog()


def custom_model(input_shape, output_shape):

	'''
	This funtion creates a Sequential model With our architecture
	for classification task

	network
	=======

	(convolution,maxpooling,dropout) * 3
	Flatten
	Dense(128)
	Dense(no of classes)


	Arguments
	=========
	input_shape  :([image_width,image_height,num_channels])
	output_shape : No of classes

	Output
	======
	Return a Sequential model instance
	'''

	model = Sequential()
	model.add(Conv2D(
			filters=32, kernel_size=(5, 5), activation="relu",
			input_shape=input_shape))
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.3))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.5))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.6))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(output_shape, activation='softmax'))
	return model


def train(num_epochs):

	'''
	This function helps us to train our model
	First we create our model then we import 
	and preprocess our data then initiate training 
	using keras fit mehod

	Arguments
	=========
	num_epochs : No of epochs which the model to be trained

	Returns
	=======
	Returns nothing save the trained model as custom_model.h5 
	'''

	model = custom_model((28, 28, 1), 10)
	model.compile(
			optimizer='rmsprop', loss='binary_crossentropy',
			metrics=['accuracy'])

	'''
	Importing and preprocessing our data.Preprocessing data is needed
	because we need to match our model's input_shape with our actual 
	inputs so some reshaping is required and it is done using numpy
	'''

	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
	x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
	y_train = np.eye(10)[y_train].astype(np.float32)
	y_test = np.eye(10)[y_test].astype(np.float32)

	'''
	Callback function to stop training when our model overfits,
	ie when the validation loss rises
	'''

	call_back = keras.callbacks.EarlyStopping(
			monitor='loss', mode="min", verbose=1)

	'''
	Train our model
	'''

	history = model.fit(
			x_train, y_train, batch_size=64,
			epochs=num_epochs, callbacks=[call_back])
	score = model.evaluate(
			x_test, y_test, batch_size=64, verbose=1)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	'''
	Saving Model
	'''
	model.save('custom_model.h5')


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--num_epochs", default=10, type=int, help="Number of epochs")
	args = parser.parse_args()
	train(num_epochs=args.num_epochs)
