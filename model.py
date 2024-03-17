from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import datasets, layers
import numpy as np
from sklearn.metrics import accuracy_score


class ANN:
    def __init__(self):
        self.datagen = None
        self.accuracy = 0
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.dataset_ready = False

        self.model = None

    def build(self):
        self.model = Sequential([
            layers.Rescaling(1. / 255),
            layers.RandomRotation(factor=(-0.2, 0.2)),
            layers.RandomZoom(height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3)),

            layers.ZeroPadding2D(padding=(1, 1)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)),
            layers.MaxPooling2D((2, 2)),

            layers.ZeroPadding2D(padding=(1, 1)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.ZeroPadding2D(padding=(1, 1)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

            layers.Flatten(),
            layers.Dense(units=64, activation='relu'),
            layers.Dense(10),
            layers.Softmax(),
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def load_dataset(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = datasets.mnist.load_data()

        self.X_train = self.X_train.reshape((60000, 28, 28, 1))
        self.X_test = self.X_test.reshape((10000, 28, 28, 1))

        encoded_array = []
        for i in self.y_train:
            arr = [0] * 10
            arr[i] = 1
            encoded_array.append(arr)
        self.y_train = np.array(encoded_array)

        self.dataset_ready = True

    def fit(self, epochs=10, batch_size=32):
        self.model.fit(
            x=self.X_train,
            y=self.y_train,
            batch_size=batch_size,
            epochs=epochs
        )

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
        self.model = load_model(path)
