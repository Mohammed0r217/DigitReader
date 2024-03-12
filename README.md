# Digit Reader

A program capable of reading single digits written by the user.

## Technologies Used

- Python 3.11.3
- Pygame 2.4.0
- TensorFlow 2.12.0

## Setup and Installation

1. Clone this repository to your local machine.
2. Install the necessary packages:
```bash
pip install pygame tensorflow
```
3. Navigate to the repository's directory and run the main file to start the program:
```bash
python main.py
```

## How It Works

The program leverages a Convolutional Neural Network (CNN) trained on the MNIST dataset, a large database of handwritten digits. Users can interactively draw single digits using their cursor. Upon pressing the ‘read’ button, the drawn digit is fed into the trained network. The program then interprets the digit and provides the corresponding numerical output.

## License

This project is licensed under the terms of the MIT license.
