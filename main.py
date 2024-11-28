import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# User inputs with defaults
train_new_model = input("Train a new model? (yes/no) [default: yes]: ").strip().lower()
train_new_model = train_new_model == 'yes' if train_new_model else True

if train_new_model:
    num_layers = input("Enter the number of layers [default: 2]: ").strip()
    num_layers = int(num_layers) if num_layers else 2

    neurons_per_layer = input("Enter the number of neurons per layer [default: 128]: ").strip()
    neurons_per_layer = int(neurons_per_layer) if neurons_per_layer else 128

    activation = input("Enter activation function (relu, sigmoid, tanh, etc.) [default: relu]: ").strip()
    activation = activation if activation else 'relu'

    epochs = input("Enter the number of epochs [default: 5]: ").strip()
    epochs = int(epochs) if epochs else 5

if train_new_model:
    # Loading the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network dynamically based on user input
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(units=neurons_per_layer, activation=activation))
    
    # Output layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Compile, optimize and train a new model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")

    model.save('handwritten_digits.keras')
else:
    model = tf.keras.models.load_model('handwritten_digits.keras')

# Note. Images should ideally be 28x28
image_number = 1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_number}.png')[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except Exception as e:
        print(f"Error reading image! Proceeding with next image... ({e})")
        image_number += 1
