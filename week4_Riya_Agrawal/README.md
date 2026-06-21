CIFAR-10 Image Classification: ANN vs CNN

Overview
This project compares Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) for image classification using the CIFAR-10 dataset. The goal is to understand why CNNs perform better on image data and analyze the impact of training strategies such as Dropout, Batch Normalization, and Data Augmentation.

Dataset
- CIFAR-10 Dataset
- 60,000 color images (32×32×3)
- 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

Models Implemented
ANN
- Dense (512) → Dropout → Dense (256) → Output
- Test Accuracy: **41.57%**

CNN
- Conv2D → BatchNorm → MaxPool
- Conv2D → BatchNorm → MaxPool
- Conv2D → Dense → Dropout → Output
- Test Accuracy: **70.68%**

Results
| Model  | Test Accuracy |
|--------|--------------|
| ANN    | 41.57% |
| CNN    | 70.68% |

Techniques Used
- Normalization
- Dropout
- Batch Normalization
- Data Augmentation
- Early Stopping (optional)

Technologies
- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib

Conclusion
CNN significantly outperformed ANN by preserving spatial information and learning image features effectively. Training strategies further improved model generalization and performance.
