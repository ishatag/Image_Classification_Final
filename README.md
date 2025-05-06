# Image_Classification_Final: Cats vs Dogs (CNN using Transfer Learning)
A deep learning project that classifies images as either a cat 🐱 or a dog 🐶 using Convolutional Neural Networks (CNNs) with Transfer Learning via the VGG16 model.

📌Project Objective
The goal of this project is to build a binary image classifier that distinguishes between cats and dogs with high accuracy. This demonstrates my skills in:

1. Working with image data
2. Building and training deep learning models
3. Evaluating model performance
4. Making real-world predictions

🧰 Tools & Technologies
| Tool/Library       | Purpose                                |
| ------------------ | -------------------------------------- |
| Python             | Programming language                   |
| TensorFlow & Keras | Deep learning framework                |
| VGG16              | Pre-trained CNN for feature extraction |
| Matplotlib         | Visualizing model accuracy/loss        |
| NumPy              | Array operations                       |
| Kaggle Notebooks   | Cloud-based development environment    |


🔄 Workflow Overview
1. Data Preprocessing:
a)Loaded cat and dog images from directory.
b)Resized all images to 224x224 pixels.
c)Normalized pixel values to [0, 1].

2. Model Architecture:
a)Used VGG16 (without top layers) as the base model.
b)Added:
(i)GlobalAveragePooling2D
(ii)Dense(256, activation='relu')
(iii)Dropout(0.5)
(iv)Dense(1, activation='sigmoid')

4. Training:
a)Optimizer: Adam
b)Loss function: BinaryCrossentropy
c)Epochs: 10
d)Metrics tracked: Accuracy

5. Evaluation & Prediction:
a)Evaluated on validation dataset
b)Uploaded external images (e.g., guesswho.jpg and guessthe animal) to test real-world predictions

📈 Results
| Metric                         | Value                                                         |
| ------------------------------------- | ------------------------------------------------------ |
| **Validation Accuracy**               | \81%                                                  |
| **Validation Loss**                   | Low and stable (did not overfit)                       |
| **Prediction on guesswho.jpg**        | ✅ Correctly predicted: `Dog`                         | 
| **Prediction on guesstheanimal.jpg**  | ✅ Correctly predicted: `Cat`                         |
| **Model Size**                        | Lightweight due to use of `include_top=False` in VGG16 |


🖼️ Real-World Prediction Examples
Uploaded images:

guesswho.jpg

✅ Correctly predicted: Dog 🐶

Confidence: ~97%

guesstheanimal.jpg

✅ Correctly predicted: Cat 🐱 

Confidence: ~67% 

➡️ These tests demonstrate the model’s ability to generalize and classify previously unseen images with high confidence.






