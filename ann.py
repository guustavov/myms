from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

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

class SoftmaxANN(object):
    def __init__(self):
        self.model = Sequential()

    def fit(self, x, Y):
        # Represent class label as binary vectors (One Hot Encoding)
        Y = to_categorical(Y)

        inputNeurons = 23
        hiddenNeurons = 23
        outputNeurons = 2
        hiddenActivation = 'relu'
        outputActivation = 'softmax'

        self.model.add(Dense(inputNeurons, input_dim = inputNeurons))
        self.model.add(Dense(hiddenNeurons, kernel_initializer='normal', activation=hiddenActivation))
        self.model.add(Dense(outputNeurons, kernel_initializer='normal', activation=outputActivation))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='loss',patience=20)
        return self.model.fit(x, Y, epochs=500, verbose=2, callbacks=[early_stopping])

    def predict(self, test_x):
        return self.model.predict(test_x)

    def predict_classes(self, test_x):
        return self.model.predict_classes(test_x)

    def getName(self):
        return self.__class__.__name__