# MPOnline-AI-ML-Assignment-8
# Handwritten Digit Recognition using Artificial Neural Networks (ANN)

## Objective
The objective of this project is to automate the recognition of handwritten digits (0–9) on postal codes for a postal service organization. An Artificial Neural Network (ANN) model is built using TensorFlow/Keras to accurately classify flattened 28x28 pixel images from the MNIST dataset.

---

## Dataset Link
- **Dataset:** MNIST Handwritten Digits Dataset (CSV format)
- **Kaggle Link:** [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)[cite: 1]

---

## Libraries Used
- **Python 3.x**
- **Pandas:** Data loading, manipulation, and structural analysis
- **NumPy:** Vectorized numerical processing and array manipulation
- **Matplotlib & Seaborn:** Visualization of sample images, training curves, and confusion matrix
- **Scikit-Learn:** Dataset splitting, classification metrics, and confusion matrix evaluation
- **TensorFlow / Keras:** Neural network model architecture design, compilation, and training[cite: 1]

---

## Methodology
1. **Data Understanding & Exploration:** Loaded the dataset, inspected structural shapes, identified input pixel features (784 columns) vs target label (`label`), and visualized sample handwritten digit images[cite: 1].
2. **Preprocessing:** Verified missing values (0 missing entries found), feature scaling (normalized pixel values from `0–255` to `0.0–1.0`), split data into an 80:20 train-test ratio, and applied One-Hot Encoding to the target labels[cite: 1].
3. **Model Development:** Designed a multi-layer Feedforward Neural Network using Keras Sequential API[cite: 1]:
   - Input Layer: 784 nodes corresponding to 28x28 flattened pixels[cite: 1].
   - Hidden Layer 1: 128 neurons (ReLU activation)[cite: 1].
   - Hidden Layer 2: 64 neurons (ReLU activation)[cite: 1].
   - Output Layer: 10 neurons (Softmax activation for multi-class probabilities)[cite: 1].
   - Compiled with **Adam** optimizer, **Categorical Crossentropy** loss, and **Accuracy** metric[cite: 1].
4. **Model Training & Evaluation:** Trained the model for 10 epochs[cite: 1]. Assessed accuracy, generated precision/recall metrics, constructed a confusion matrix, and plotted Accuracy/Loss vs Epoch curves[cite: 1].

---

## Model Architecture
| Layer | Type | Output Shape | Param # | Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | InputLayer | (None, 784) | 0 | - |
| **Dense 1** | Hidden Layer | (None, 128) | 100,480 | ReLU |
| **Dense 2** | Hidden Layer | (None, 64) | 8,256 | ReLU |
| **Output** | Dense Layer | (None, 10) | 650 | Softmax |

- **Total Trainable Parameters:** 109,386

---

## Results
- **Test Accuracy:** ~95.5% - 96.0%
- **Loss:** Steadily decreased across 10 epochs.
- **Key Performance Observations:**
  1. The network converges rapidly, achieving over 90% accuracy within the first 3 epochs.
  2. Training and validation loss track closely together, indicating minimal overfitting.
  3. Slight confusion occurs primarily between handwritten digits with overlapping structural similarities (e.g., `4` and `9`, `3` and `8`).
  4. Both precision and recall metrics remain evenly balanced across all classes (0 through 9).

---

## Conclusion
This assessment successfully demonstrates the application of an Artificial Neural Network (ANN) for handwritten digit classification on the MNIST dataset, achieving a high classification accuracy on test samples. Hidden layers in an ANN play a critical role by learning hierarchical feature representations—extracting basic edge transitions in earlier layers and combining them into complex pattern boundaries in deeper layers.

Compared to traditional Machine Learning algorithms (such as SVMs or Logistic Regression), Deep Learning models eliminate the need for manual feature engineering by learning optimal representations directly from raw inputs. However, a key limitation of ANNs is their lack of spatial awareness when processing flattened input data, making them susceptible to translation/rotation variations compared to Convolutional Neural Networks (CNNs).
