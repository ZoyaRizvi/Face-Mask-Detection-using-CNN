# Face Mask Detection using Convolutional Neural Networks

## Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images as **with mask** or **without mask**. The model is trained on a labeled image dataset and demonstrates strong performance in face mask detection tasks.

## Dataset

The dataset used is the **Face Mask Dataset** from Kaggle.

* **Source**: `omkargurav/face-mask-dataset`
* **Classes**:

  * `with_mask`
  * `without_mask`

## Tech Stack

* Python
* TensorFlow, Keras
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib
* Pillow

## Methodology

### Data Preprocessing

* Images are loaded from both classes and resized to **128 × 128** pixels.
* Images are converted to RGB format.
* Pixel values are normalized to the range **[0, 1]**.
* Labels are assigned (`1` for mask, `0` for no mask).
* Dataset is split into **80% training** and **20% testing** data.

### Model Architecture

The CNN is built using the **Keras Sequential API** and includes:

* Convolutional layers with ReLU activation
* Max pooling layers for feature reduction
* Fully connected dense layers
* Dropout layers to reduce overfitting
* Sigmoid-activated output layer for binary classification

### Training

* **Optimizer**: Adam
* **Loss Function**: Sparse Categorical Crossentropy
* **Metric**: Accuracy
* **Epochs**: 5

## Results

* The model achieves a **test accuracy of approximately 92%**.
* Training and validation accuracy and loss are visualized to evaluate model performance.

## How to Run the Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/face-mask-detection-cnn.git
   cd face-mask-detection-cnn
   ```

2. **Install dependencies**

   ```bash
   pip install numpy matplotlib opencv-python pillow scikit-learn tensorflow keras kaggle
   ```

3. **Download the dataset**

   * Download the dataset from Kaggle: `omkargurav/face-mask-dataset`
   * Extract it into the project directory maintaining the folder structure:

     ```
     dataset/
       ├── with_mask/
       └── without_mask/
     ```

4. **Run the notebook or script**

   * Train the CNN model
   * Evaluate performance on test data
   * Save the trained model (optional)

## Prediction

To predict on a new image:

1. Load and resize the image to **128 × 128**
2. Normalize pixel values
3. Reshape the image to match the model input
4. Use `model.predict()` to classify the image

The output indicates whether the person is **wearing a mask** or **not wearing a mask**.

## Conclusion

This project demonstrates the application of **Convolutional Neural Networks** for image classification and serves as a strong foundation for real-world computer vision tasks such as public safety monitoring and real-time detection systems.
