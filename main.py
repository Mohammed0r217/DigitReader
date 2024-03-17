import myappkit
import numpy as np
import model
import drawing_board
from os.path import isfile

app = myappkit.App(title='Digit Recognizer', w=538, h=392)

home = myappkit.Activity(bgColor=(0, 0, 0), frame_rate=120)
home.addItem(drawing_board.DrawingBoard())

digitRecgnizer = model.ANN()


def train_model():
    if digitRecgnizer.model is None:
        digitRecgnizer.build()

    if not digitRecgnizer.dataset_ready:
        digitRecgnizer.load_dataset()

    digitRecgnizer.fit(epochs=10)
    digitRecgnizer.save('model.keras')


def load_model():
    digitRecgnizer.load('model.keras')


def read_digit():
    if digitRecgnizer.model is None:
        if isfile('model.keras'):
            load_model()
        else:
            return

    x = home.items[0].get_digit()
    x = np.reshape(x, (1, 28, 28, 1))

    y_pred = digitRecgnizer.predict(x)
    digit = np.argmax(y_pred)
    home.items[0].digit = digit
    home.items[0].probability = round(y_pred[0][digit]*100)

    """plt.imshow(x[0], cmap='gray')
    plt.show()"""


home.addItem(myappkit.Button(
    rect=(0, 0, 200, 60),
    text='Train Model',
    onClick=train_model
))

home.addItem(myappkit.Button(
    rect=(0, 61, 200, 60),
    text='Load Model',
    onClick=load_model
))

home.addItem(myappkit.Button(
    rect=(0, 61*2, 200, 60),
    text='Read',
    onClick=read_digit
))

home.addItem(myappkit.Button(
    rect=(0, 61*3, 200, 60),
    text='Clear',
    onClick=home.items[0].clear
))

app.activities['home'] = home

app.run()
