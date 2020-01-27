import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

class OriginalANN(object):
	def __init__(self):
		self.model = Sequential()

	def fit(self, x, Y):
		inputNeurons = 23
		hiddenNeurons = 23
		outputNeurons = 1
		inputActivation = hiddenActivation = outputActivation = 'tanh'

		self.model.add(Dense(inputNeurons, input_dim = inputNeurons, kernel_initializer='normal', activation=inputActivation))
		self.model.add(Dense(hiddenNeurons, kernel_initializer='normal', activation=hiddenActivation))
		self.model.add(Dense(outputNeurons, kernel_initializer='normal', activation=outputActivation))

		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		early_stopping = EarlyStopping(monitor='loss',patience=20)
		return self.model.fit(x, Y, epochs=500, verbose=0, callbacks=[early_stopping])

	def predict(self, test_x):
		return self.model.predict(test_x)

	def predict_classes(self, test_x):
		return self.model.predict_classes(test_x)

	def getName(self):
		return self.__class__.__name__

	def setAnnModel(self, annModel):
		self.model = annModel

	def clearModel(self):
		self.model = Sequential()

class SoftmaxANN(object):
	def __init__(self):
		self.model = Sequential()
		self.alreadyTrained = False
		pass

	def fit(self, x, Y, numberOfHiddenLayers = 1):
		if self.alreadyTrained:
			return None

		# Represent class label as binary vectors (One Hot Encoding)
		one_hot_Y = to_categorical(Y)

		# number of columns (features) of the data
		inputNeurons = x.shape[1]
		# once its one hot encoded, the length each element in Y is the number of classes
		outputNeurons = len(one_hot_Y[0])

		hiddenActivation = 'relu'
		outputActivation = 'softmax'

		self.model.add(Dense(inputNeurons, input_dim = inputNeurons))
		self.model.add(Dense(1024, activation=hiddenActivation))
		self.model.add(Dropout(0.01))
		
		if numberOfHiddenLayers > 1:
			self.model.add(Dense(768, activation=hiddenActivation))
			self.model.add(Dropout(0.01))
		
		if numberOfHiddenLayers > 2:
			self.model.add(Dense(512, activation=hiddenActivation))
			self.model.add(Dropout(0.01))
		
		if numberOfHiddenLayers > 3:
			self.model.add(Dense(256, activation=hiddenActivation))
			self.model.add(Dropout(0.01))
		
		if numberOfHiddenLayers > 4:
			self.model.add(Dense(128, activation=hiddenActivation))
			self.model.add(Dropout(0.01))

		self.model.add(Dense(outputNeurons, activation=outputActivation))

		optimizer = Adam(lr=0.01)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

		early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)

		if outputNeurons > 2: # that is, it's a multiclass classification
			class_weight = compute_class_weight('balanced', np.unique(Y), Y)
			return self.model.fit(x, one_hot_Y, epochs=200, batch_size=2048, validation_split=0.1, shuffle=True, verbose=2, callbacks=[early_stopping], class_weight=class_weight)

		return self.model.fit(x, one_hot_Y, epochs=200, batch_size=2048, validation_split=0.1, shuffle=True, verbose=2, callbacks=[early_stopping])

	def predict(self, test_x):
		return np.argmax(self.model.predict(test_x), axis = 1)

	def predict_classes(self, test_x):
		return self.model.predict_classes(test_x)

	def getName(self):
		return self.__class__.__name__

	def setAnnModel(self, annModel):
		self.model = annModel
		self.alreadyTrained = True

	def clearModel(self):
		self.model = Sequential()
		self.alreadyTrained = False