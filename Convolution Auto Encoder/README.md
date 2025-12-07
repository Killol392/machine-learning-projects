# Convolutional Autoencoder – CIFAR-10

This project implements a convolutional autoencoder on the CIFAR-10 dataset (60,000 color images, 10 balanced classes). The focus is on feature learning, reconstruction quality, and understanding latent representations for textured RGB images.

## 1. Dataset & Preprocessing
- CIFAR-10 shapes:
  - **Train:** (50000, 32, 32, 3)  
  - **Test:** (10000, 32, 32, 3)
- Pixel values converted to **float32** and normalized to **[0,1]**.
- Normalization stabilizes gradients; training without it causes slow convergence and unstable reconstructions.
- No heavy augmentation used - autoencoders must reconstruct exact inputs.

```
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
```

## 2. EDA Summary
- Images are **32×32 RGB**, containing real textures (fur, water, grass), making them more complex than MNIST.
- Wide variation in color, contrast, and background clutter.
- All 10 classes are evenly distributed (6k images each).
- Random 25-image grid shows high intra-class variability and background noise, explaining why deeper models are required.

## 3. Model Architecture
### Encoder
- 3 × (Conv -> BatchNorm -> ReLU -> MaxPool)
- Feature depth increases: **64 -> 128 -> 256 -> 512**
- Final latent representation: **4×4×512**

### Decoder  
- Symmetric: 3 × (ConvTranspose -> BatchNorm -> ReLU)
- Final Conv layer with **sigmoid** outputs a 32×32×3 reconstructed image.

### Full Model
- ~ **3.1M parameters**
- Loss: **MSE**  
- Optimizer: **Adam**  
- Epochs: **40**, batch size 128

## 4. Training Results
- Training loss: **0.017 -> 0.0014**
- Validation loss: **0.021 -> 0.0017**
- Smooth, parallel loss curves -> no overfitting.

## 5. Reconstruction Results
### Qualitative
- Reconstructed images retain:
  - global shapes  
  - dominant colors  
  - object boundaries  
- Fine textures (fur, metal edges) appear slightly blurred due to compression.

### Quantitative
- **Test MSE:** 0.001893  
- **PSNR:** ~27.7 dB  
- Matches expected quality for 32×32 natural images.

## 6. Key Technical Insights
### Why deeper Conv layers?
CIFAR-10 has:
- textures  
- color gradients  
- cluttered backgrounds  
which require **multiple convolution stages** and **high-depth feature maps**. MNIST-level networks would underfit heavily.

### Color images -> higher model capacity
- RGB triples the input dimensionality.
- First Conv layer alone has 3× more weights than in grayscale datasets.
- More parameters -> higher risk of **overfitting**, mitigated by BatchNorm, pooling, and dataset size.

### Latent representation differences
- **Color images:** encode hue, saturation, texture, and spatial structure -> require high-dimensional latent tensors.
- **Grayscale images:** only encode shape and intensity -> much smaller latent spaces suffice.

## 7. Reflection
Working with CIFAR-10 autoencoders highlighted the need for deeper architectures, careful normalization, and stabilizing layers like BatchNorm. The project reinforced how textures and color drastically increase representational complexity, and how reconstruction metrics (MSE, PSNR) provide a grounded understanding of model performance.

