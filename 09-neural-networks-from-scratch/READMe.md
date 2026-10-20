# Neural Networks from Scratch

## Overview

This section introduces the fundamental concepts of artificial neural networks through hands-on implementations using NumPy.

The goal is to understand how neural networks work internally before moving to more advanced architectures and deep learning frameworks in the Deep Learning Portfolio.

Instead of relying on automatic differentiation or prebuilt neural network models, the core components are implemented manually, including forward propagation, loss functions, backpropagation, and gradient descent.

Two complementary projects are covered:

1. **Neural Network Fundamentals:** Build a Perceptron and a Multilayer Perceptron (MLP) from scratch, using logical classification and XOR to understand nonlinear decision boundaries.
2. **MLP Digits Classification:** Apply the same principles to a real handwritten-digit classification dataset and evaluate the trained model on unseen data.

## Project Structure

```text
09-neural-networks-from-scratch/
├── 01_neural_network_fundamentals.ipynb
├── 02_mlp_digits_classification.ipynb
└── README.md
```

| Notebook | Description |
|---|---|
| `01_neural_network_fundamentals.ipynb` | Perceptron, linear separability, MLP, forward propagation, loss functions, gradient descent, backpropagation, and XOR classification. |
| `02_mlp_digits_classification.ipynb` | Multiclass classification using a NumPy-based MLP, the Digits dataset, training, evaluation, and error analysis. |

## Learning Objectives

By completing this section, the following concepts are explored through practical implementation:

- Understand neurons, weights, biases, and activation functions.
- Implement a Perceptron and examine the limitations of linear classifiers.
- Understand why hidden layers and nonlinear activations are necessary for solving XOR.
- Implement forward propagation using matrix operations.
- Understand Binary Cross-Entropy and Multiclass Cross-Entropy.
- Implement Softmax for multiclass classification.
- Derive and implement gradients using backpropagation.
- Validate analytical gradients using numerical finite differences.
- Train neural networks using gradient descent.
- Evaluate classification performance using accuracy, precision, recall, F1-score, and a confusion matrix.

---

## 01 — Neural Network Fundamentals

### Objective

Build a basic neural network from scratch and understand its internal learning mechanism through a series of progressively more complex experiments.

### Topics Covered

**1. Perceptron**

- Inputs, weights, and bias.
- Weighted sums and threshold-based predictions.
- The Perceptron learning rule.
- Training on the AND logical operation.

**2. Linear Separability and XOR**

The XOR problem demonstrates the limitations of a single-layer Perceptron. Its classes cannot be separated by a single linear decision boundary.

This motivates the introduction of hidden layers and nonlinear activation functions.

**3. Multilayer Perceptron**

Implement a network with the architecture:

```text
Input Layer → Hidden Layer → Output Layer
   2       →       2       →      1
```

The implementation covers:

- Forward propagation.
- Sigmoid activation.
- Binary Cross-Entropy loss.
- Analytical and numerical gradient comparison.
- Backpropagation using the chain rule.
- Gradient descent and parameter updates.
- Training-loop implementation.

### XOR Results

The trained MLP successfully learned the XOR mapping.

| Input | Target | Predicted Class |
|---|---:|---:|
| `[0, 0]` | 0 | 0 |
| `[0, 1]` | 1 | 1 |
| `[1, 0]` | 1 | 1 |
| `[1, 1]` | 0 | 0 |

Final results after 10,000 training epochs:

| Metric | Result |
|---|---:|
| Initial Loss | 0.6994 |
| Final Loss | 0.000986 |
| Training Accuracy | 100% |

The experiment demonstrates that a small MLP with a nonlinear hidden layer can learn a decision boundary that a single linear classifier cannot represent.

---

## 02 — MLP Digits Classification

### Objective

Apply the neural network fundamentals to a real classification task using handwritten-digit images.

The model is implemented using NumPy rather than a deep learning framework.

### Dataset

The project uses the Digits dataset provided by scikit-learn.

| Property | Value |
|---|---|
| Total samples | 1,797 |
| Image dimensions | 8 × 8 pixels |
| Input features | 64 |
| Number of classes | 10 |
| Class labels | 0–9 |
| Pixel intensity range | 0–16 |

The dataset is divided into training and test sets using an 80/20 split, with stratification to preserve class proportions.

| Dataset | Samples |
|---|---:|
| Training set | 1,437 |
| Test set | 360 |

Pixel values are normalized to the range `[0, 1]`, and class labels are converted to one-hot representations for training.

### Network Architecture

The implemented MLP uses the following architecture:

```text
Input Layer       Hidden Layer       Output Layer
  64 neurons  →    64 neurons   →     10 neurons
                      ReLU             Softmax
```

| Component | Implementation |
|---|---|
| Input representation | 64 normalized pixel features |
| Hidden layer | 64 neurons |
| Hidden activation | ReLU |
| Output layer | 10 neurons |
| Output activation | Softmax |
| Loss function | Multiclass Cross-Entropy |
| Optimization | Full-Batch Gradient Descent |
| Learning rate | 0.1 |
| Training epochs | 1,000 |
| Parameter initialization | He initialization for the hidden layer |

### Implementation Details

The network is implemented from scratch, including:

1. Weight and bias initialization.
2. Forward propagation through the hidden and output layers.
3. Softmax probability computation with numerical stabilization.
4. Multiclass Cross-Entropy loss calculation.
5. Backpropagation through the output and hidden layers.
6. Gradient Descent parameter updates.
7. Training-loss and accuracy tracking.
8. Evaluation on unseen test samples.
9. Confusion matrix and classification-report analysis.

The analytical gradients are checked against numerical finite-difference estimates to validate the backpropagation implementation.

### Final Results

The model achieved the following performance after 1,000 training epochs.

| Metric | Result |
|---|---:|
| Training Loss | 0.0907 |
| Training Accuracy | 98.12% |
| Test Loss | 0.1277 |
| Test Accuracy | 96.39% |
| Macro Precision | 96.50% |
| Macro Recall | 96.36% |
| Macro F1-score | 96.35% |
| Correct test predictions | 347 / 360 |
| Incorrect test predictions | 13 / 360 |

The model correctly classified 347 of the 360 test samples, achieving a test accuracy of approximately 96.39%.

The relatively small difference between training and test accuracy is encouraging, although a separate validation set would be needed for more systematic model selection and hyperparameter tuning.

### Error Analysis

The confusion matrix and misclassified images provide additional insight beyond overall accuracy.

Key observations:

- Digits 2, 4, and 5 achieved 100% recall on this test set.
- Digit 8 was the most challenging class, with a recall of 85.71%.
- Some handwritten digits were visually ambiguous at the dataset's 8 × 8 resolution.
- The model occasionally confused digits with similar visual structures, including 8 with 1 or 7, and 9 with 4.

These findings demonstrate why per-class metrics and error inspection are useful when evaluating classification models.

---

## Technologies Used

- **Python** — implementation language.
- **NumPy** — matrix operations, neural network calculations, and manual gradient computation.
- **Matplotlib** — training curves, decision boundaries, and visualizations.
- **scikit-learn** — Digits dataset, train-test splitting, confusion matrix, and classification metrics.
- **Jupyter Notebook** — interactive implementation and experimentation.

PyTorch and automatic differentiation are not used to train the networks in these exercises.

## Installation and Execution

Install the required packages:

```bash
pip install numpy matplotlib scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in the following order:

1. `01_neural_network_fundamentals.ipynb`
2. `02_mlp_digits_classification.ipynb`

Each notebook contains the implementation, experiments, visualizations, and analysis for its respective topic.

## Limitations and Scope

This section focuses on understanding neural network fundamentals rather than building a production-ready deep learning system.

The current implementations have several limitations:

- The XOR experiment uses only four samples.
- The Digits dataset contains small, low-resolution grayscale images.
- The Digits classifier uses a single hidden layer.
- Only one primary architecture and set of training hyperparameters were evaluated.
- Full-batch Gradient Descent is used instead of mini-batch or stochastic optimization.
- No separate validation set or systematic hyperparameter search is included.
- The reported results correspond to the specific dataset split and initialization used in the notebook.

These limitations are appropriate for an introductory implementation-focused project.

## Conclusion

This section establishes a practical understanding of how neural networks learn, from a single Perceptron to a multilayer classifier.

By implementing the fundamental components manually, the project demonstrates how forward propagation produces predictions, how loss functions measure prediction errors, how backpropagation calculates gradients, and how gradient descent updates parameters to improve performance.

The XOR experiment illustrates the importance of nonlinear representations, while the Digits project demonstrates that the same underlying principles can be applied to a real multiclass classification task.

This foundation prepares the way for more advanced neural network architectures and deep learning methods in the Deep Learning Portfolio.
