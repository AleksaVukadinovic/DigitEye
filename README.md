# DigitEye

This project implements a neural network using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset. The model can also predict custom handwritten digit images.

---

## Features
- **Customizable Neural Network:** Dynamically set the number of layers, neurons, activation functions, and epochs.
- **Data Augmentation:** Increases the training dataset using techniques like rotation, shifting, and zooming to improve generalization.
- **Custom Predictions:** Load your custom digit images and let the model predict them.
- **Model Saving and Loading:** Train a new model or load an existing one.

---

## Installation

### Prerequisites
- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

Install the required libraries with:

```bash
pip install tensorflow numpy opencv-python matplotlib
```

Note that if you are using Linux, you will most likely need to create virtual environment to install these libraries, as pip installing is no longer recommended.

## Materials used

- [DeepLearning series by 3Blue1Brown] (https://www.youtube.com/watch?v=aircAruvnKk list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Handwritten Digit Recognition by NeuralNine] (https://youtu.be/bte8Er0QhDg?si=Nf-ChxRHdiWNV2M0)
- [Veštačka inteligencija book by Mladen Nikolić & Predrag Janičić] (https://poincare.matf.bg.ac.rs/~janicic//books/VI_B5.pdf)
- [Using neural networks to recognize digits article] (http://neuralnetworksanddeeplearning.com/chap1.html)