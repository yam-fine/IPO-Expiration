from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from tensorflow.keras import regularizers
from matplotlib import pyplot
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


DATES_BACK = 3
COLUMNS_LEAVE_ME_ALONE_DECREASE_ME_THERE = [0, 1, 3, 9]
NUMBER_OF_FEATURES = 5

class learningAlgs():
    def __init__(self, X, y):
        self.X = np.asarray(X.to_numpy()).astype(np.float32)
        self.y = np.asarray(y.to_numpy()).astype(np.float32)

    #two layer simple nn with MSE
    def neuralNetMSE(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
        # define the keras model
        model = Sequential()
        model.add(Dense((DATES_BACK + 1) * NUMBER_OF_FEATURES, input_dim= DATES_BACK * NUMBER_OF_FEATURES, activation = 'sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='softmax'))
        opt = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=opt)
        # fit model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
        # evaluate the model
        train_mse = model.evaluate(X_train, y_train, verbose=0)
        test_mse = model.evaluate(X_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
        # plot loss during training
        pyplot.title('Loss / Mean Squared Error')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    #two layer simple nn with MSLE
    def neuralNetMSLE(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
        # define the keras model
        model = Sequential()
        model.add(Dense(DATES_BACK * NUMBER_OF_FEATURES, input_dim=DATES_BACK * NUMBER_OF_FEATURES, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='softmax'))
        opt = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
        model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])
        # fit model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
        # evaluate the model
        train_mse = model.evaluate(X_train, y_train, verbose=0)
        test_mse = model.evaluate(X_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_mse[0], test_mse[0]))
        # plot loss during training
        pyplot.title('Loss / Mean Squared Error')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    #two layer simple nn with MAE
    def neuralNetMAE(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
        # define the keras model
        model = Sequential()
        model.add(Dense(DATES_BACK * NUMBER_OF_FEATURES, input_dim=DATES_BACK * NUMBER_OF_FEATURES, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear'))
        opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
        model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mse'])
        # fit model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
        # evaluate the model
        train_mse = model.evaluate(X_train, y_train, verbose=0)
        test_mse = model.evaluate(X_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_mse[0], test_mse[0]))
        # plot loss during training
        pyplot.title('Loss / Mean Squared Error')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    #multi layer "medium" nn with MSE
    def neuralNetMSELarge(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
        # define the keras model
        model = Sequential()
        model.add(Dense((DATES_BACK + 1) * NUMBER_OF_FEATURES, input_dim=DATES_BACK * NUMBER_OF_FEATURES, activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense((DATES_BACK - 1) * NUMBER_OF_FEATURES, activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense((DATES_BACK - 2) * NUMBER_OF_FEATURES, activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear'))
        opt = SGD(lr=0.00001, momentum=0.9, decay=0.0, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=opt)
        # fit model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000, verbose=0)
        # evaluate the model
        train_mse = model.evaluate(X_train, y_train, verbose=0)
        test_mse = model.evaluate(X_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
        # plot loss during training
        pyplot.title('Loss / Mean Squared Error')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    # multi layer "medium" nn with bin classifier
    def neuralNetBinClassification(self):
        from sklearn.datasets import make_circles
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import SGD
        from matplotlib import pyplot
        treshhold = 1
        # generate 2d classification dataset
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                            random_state=30)
        y_train[np.where(y_train >= treshhold)] = 1
        y_train[np.where(y_train < treshhold)] = 0

        y_test[np.where(y_test >= treshhold)] = 1
        y_test[np.where(y_test < treshhold)] = 0

        # define model
        model = Sequential()
        model.add(Dense((DATES_BACK+1) * NUMBER_OF_FEATURES, input_dim=DATES_BACK * NUMBER_OF_FEATURES, activation='sigmoid',kernel_initializer='he_uniform'))
        model.add(Dense((DATES_BACK - 1) * NUMBER_OF_FEATURES, activation='sigmoid',kernel_initializer='he_uniform'))
        # model.add(Dense((DATES_BACK - 2) * NUMBER_OF_FEATURES, activation='sigmoid',kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
        # fit model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0)
        # evaluate the model
        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        # plot accuracy during training
        pyplot.subplot(212)
        pyplot.title('Accuracy')
        pyplot.plot(history.history['accuracy'], label='train')
        pyplot.plot(history.history['val_accuracy'], label='test')
        pyplot.legend()
        pyplot.show()
        return model
