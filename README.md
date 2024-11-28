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

Note for Linux users:
It is recommended to create a virtual environment to install these libraries, as direct pip installation may lead to system-wide conflicts. Use the following commands to set up a virtual environment:

```bash
python3 -m venv digit_eye_env
source digit_eye_env/bin/activate
pip install tensorflow numpy opencv-python matplotlib
```

## Usage
1. Clone the repository
```bash
git clone https://github.com/AleksaVukadinovic/DigitEye
cd DigitEye
```

2. Run the script
```bash
python digit_recognition.py
```

3. Follow the console prompts

## Materials Used

This project draws inspiration from and builds upon the following resources:

- [DeepLearning Series by 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  
  An excellent visual introduction to the concepts behind neural networks and deep learning.

- [Handwritten Digit Recognition by NeuralNine](https://youtu.be/bte8Er0QhDg?si=Nf-ChxRHdiWNV2M0)  
  A practical guide to implementing a digit recognition system.

- [Veštačka inteligencija (Artificial Intelligence)](https://poincare.matf.bg.ac.rs/~janicic//books/VI_B5.pdf)  
  A comprehensive textbook by Mladen Nikolić and Predrag Janičić, providing theoretical insights into artificial intelligence.

- [Using Neural Networks to Recognize Digits](http://neuralnetworksanddeeplearning.com/chap1.html)  
  A detailed article explaining the mathematics and algorithms behind digit recognition using neural networks.
