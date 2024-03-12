from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.metrics import accuracy_score


class ANN:
    def __init__(self):
        self.accuracy = 0
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.dataset_ready = False

        self.model = None

    def build(self):
        self.model = models.Sequential()

        self.model.add(layers.ZeroPadding2D(padding=(1, 1)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.ZeroPadding2D(padding=(1, 1)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.ZeroPadding2D(padding=(1, 1)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))

        self.model.add(layers.Dense(10))
        self.model.add(layers.Softmax())

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def load_dataset(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = datasets.mnist.load_data()

        self.X_train = self.X_train.reshape((60000, 28, 28, 1))
        self.X_test = self.X_test.reshape((10000, 28, 28, 1))

        self.X_train, self.X_test = self.X_train / 255.0, self.X_test / 255.0

        encoded_array = []
        for i in self.y_train:
            arr = [0] * 10
            arr[i] = 1
            encoded_array.append(arr)
        self.y_train = np.array(encoded_array)

        self.dataset_ready = True

    def fit(self, epoch=10, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epoch, batch_size=batch_size)

    def predict(self, X):
        y_pred = self.model.predict(X, verbose=0)
        return y_pred

    def evalute(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)

        self.accuracy = accuracy_score(self.y_test, y_pred)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = models.load_model(path)
