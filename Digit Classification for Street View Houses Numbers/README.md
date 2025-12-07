# SVHN Digit Classification – CNN & AlexNet

This project builds and evaluates deep learning models (Simple CNN and AlexNet) for digit classification on the **Street View House Numbers (SVHN)** dataset, a benchmark of over 600k real-world 32×32 RGB digit images extracted from Google Street View.

## 1. Dataset Overview
- **Train:** 65,931 samples  
- **Validation:** 7,326 samples  
- **Test:** 26,032 samples  
- **Image size:** 32×32 RGB  
- **Labels:** 0–9 (SVHN uses “10” for zero -> remapped to 0)  
- Dataset loaded from `.mat` files and transposed into PyTorch format.

## 2. Preprocessing
- Converted pixel values from **uint8 -> float32** and normalized to **[0,1]**.
- Reshaped images from (N, H, W, C) -> (N, C, H, W) for PyTorch.
- Labels encoded as integer tensors.
- Created efficient `DataLoader` pipelines for train/val/test with batch size 64.

### Why Normalize?
Prevents exploding gradients, stabilizes training, and allows faster convergence.  
Unnormalized data (0–255) hurts both CNN and AlexNet performance.

## 3. Data Augmentation
Applied realistic perturbations to improve generalization:
- Random rotations (±15°)  
- Horizontal flips  
- Color jitter (brightness/contrast/saturation)  
- Random translations  
These augmentations replicate real-world distortions and reduce overfitting.

## 4. Models

### Simple CNN Architecture
- **Conv -> BN -> ReLU -> MaxPool**  
- **Conv -> BN -> ReLU -> MaxPool**  
- Flatten -> FC(128) -> Dropout(0.25) -> FC(10)  
- Lightweight, fast, effective for small image tasks.

### AlexNet (32×32 Adapted)
- Five convolutional layers: 64 -> 192 -> 384 -> 256 -> 256  
- ReLU after each conv  
- MaxPool after conv layers 1, 2, and 5  
- Classifier: Dropout -> FC(1024) -> FC(512) -> FC(10)  
- Much deeper; extracts richer features.

### Regularization Techniques Used
- Batch Normalization  
- Dropout  
- Data augmentation  
- Adaptive learning (Adam optimizer)  
- Stratified validation split  

## 5. Training Summary
- **Simple CNN:** 10 epochs, lr=0.001  
- **AlexNet:** 15 epochs, lr=0.0001  
- GPU (Colab T4) used for all runs.

Average training time per sample:
- **Simple CNN:** 0.00072 s  
- **AlexNet:** 0.00339 s  

## 6. Evaluation Results (Test Set)

### Simple CNN
- **Accuracy:** 83.59%  
- **Precision:** 0.8400  
- **Recall:** 0.8359  
- **F1-Score:** 0.8354  
- **Inference speed:** 0.000031 s/sample  

### AlexNet
- **Accuracy:** 92.22%  
- **Precision:** 0.9243  
- **Recall:** 0.9222  
- **F1-Score:** 0.9223  
- **Inference speed:** 0.000072 s/sample  

AlexNet significantly outperforms due to deeper feature extraction, stronger regularization, and better high-level representations.

## 7. Key Insights
- SVHN images are significantly more complex than MNIST (color, background clutter), requiring deeper architectures and augmentation.
- Learning rate critically affected convergence; AlexNet required a lower LR for stability.
- GPU compute reduced training time from hours to minutes.
- BatchNorm + Dropout + Augmentation provided strong overfitting control.

## 8. Reflection
This project reinforced how dataset structure (color, clutter, real-world noise) impacts model depth, preprocessing, and training stability. I gained hands-on experience balancing LR, epoch count, and regularization to achieve stable, high-accuracy models at scale. AlexNet’s performance highlighted the value of deeper architectures and disciplined tuning for complex vision datasets.
