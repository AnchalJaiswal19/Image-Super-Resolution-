#This is the model Code
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Helper function to calculate PSNR
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(1.0 / mse)

# Super-Resolution model with deeper architecture
def create_sr_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Feature Extraction
    x = layers.Conv2D(64, (9, 9), padding='same')(inputs)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    # Residual Blocks
    for _ in range(6):
        res = layers.Conv2D(64, (3, 3), padding='same')(x)
        res = layers.PReLU(shared_axes=[1, 2])(res)
        res = layers.Conv2D(64, (3, 3), padding='same')(res)
        x = layers.Add()([x, res])  # Skip Connection

    # Final Enhancement Layers
    x = layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)

    return models.Model(inputs, x)

# Function to load and preprocess images
def load_images(lr_path, hr_path):
    lr_images, hr_images = [], []

    lr_files = glob.glob(os.path.join(lr_path, '*.png'))
    hr_files = glob.glob(os.path.join(hr_path, '*.png'))

    if len(lr_files) == 0 or len(hr_files) == 0:
        print("Error: No images found in the provided directories.")
        return np.array([]), np.array([])

    for img_path in lr_files:
        lr_img = cv2.imread(img_path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB) / 255.0
        lr_images.append(lr_img.astype(np.float32))

    for img_path in hr_files:
        hr_img = cv2.imread(img_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB) / 255.0
        hr_images.append(hr_img.astype(np.float32))

    return np.array(lr_images, dtype=np.float32), np.array(hr_images, dtype=np.float32)

# Paths (Ensure these are directories, not individual files)
lr_path = r"C:\Users\hp\Desktop\datasett\dataset\Raw Data\low_res"
hr_path = r"C:\Users\hp\Desktop\datasett\dataset\Raw Data\high_res"

# Load dataset
lr_images, hr_images = load_images(lr_path, hr_path)

# Ensure dataset is not empty before proceeding
if lr_images.size == 0 or hr_images.size == 0:
    print("No images were loaded. Please check the file paths.")
    exit()

# Splitting dataset
X_train, X_test, Y_train, Y_test = train_test_split(lr_images, hr_images, test_size=0.2, random_state=42)

# Define model
model = create_sr_model(X_train.shape[1:])
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

# Evaluate model
test_loss, test_mse = model.evaluate(X_test, Y_test)
print(f"Test MSE: {test_mse}")

# Predict on a sample
sample_lr = X_test[0:1]
sample_hr = Y_test[0:1]
sr_image = model.predict(sample_lr)[0]

# Enhance Resolution using Upscaling
sr_image = cv2.resize(sr_image, (sample_hr.shape[2], sample_hr.shape[1]), interpolation=cv2.INTER_CUBIC)

# Convert image to 0-255 range before processing
sr_image_255 = (sr_image * 255).astype(np.uint8)

# **Post-processing Enhancements** #

# 1. Unsharp Masking (Better Edge Enhancement than Laplacian)
sharp_kernel = np.array([[-1, -1, -1], 
                         [-1,  9, -1], 
                         [-1, -1, -1]])
sr_image_255 = cv2.filter2D(sr_image_255, -1, sharp_kernel)

# 2. Reduce Noise using Bilateral Filter (Better than Gaussian Blur)
sr_image_255 = cv2.bilateralFilter(sr_image_255, 9, 75, 75)

# 3. Adaptive Contrast Enhancement using CLAHE (Lighter Adjustment)
sr_image_lab = cv2.cvtColor(sr_image_255, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(sr_image_lab)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduced Clip Limit
l = clahe.apply(l)
sr_image_lab = cv2.merge((l, a, b))
sr_image_255 = cv2.cvtColor(sr_image_lab, cv2.COLOR_LAB2RGB)

# Normalize the processed image back to 0-1 range
sr_image = sr_image_255.astype(np.float32) / 255.0

# Compute PSNR values
psnr_lr_hr = psnr(sample_lr[0], sample_hr[0])  # Low-Res vs High-Res
psnr_sr_hr = psnr(sr_image, sample_hr[0])  # Super-Res vs High-Res
psnr_lr_sr = psnr(sample_lr[0], sr_image)  # Low-Res vs Super-Res

print(f"PSNR Low-Res vs High-Res: {psnr_lr_hr:.2f} dB")
print(f"PSNR Super-Res vs High-Res: {psnr_sr_hr:.2f} dB")
print(f"PSNR Low-Res vs Super-Res: {psnr_lr_sr:.2f} dB")

# Convert images for visualization
sample_lr_255 = (sample_lr[0] * 255).astype(np.uint8)
sample_hr_255 = (sample_hr[0] * 255).astype(np.uint8)

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_lr_255)
plt.title("Low-Resolution Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sample_hr_255)
plt.title("High-Resolution Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sr_image_255)
plt.title("Enhanced Super-Resolution Image")
plt.axis('off')

plt.show()

# Save the super-resolved image
cv2.imwrite('super_resolved_image.png', sr_image_255)

#This Code for load Sample images
import numpy as np
import tensorflow as tf
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('super_resolution_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

print("Model loaded successfully.")

# Function to preprocess a single low-resolution image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}. Check if the file exists and is accessible.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
    return image.astype(np.float32)

# Function to enhance a single low-resolution image
def enhance_image(image_path, output_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return

    img_name = os.path.basename(image_path)

    # Load and preprocess the low-resolution image
    lr_image = preprocess_image(image_path)
    if lr_image is None:
        return  # Exit if image couldn't be read

    lr_image = np.expand_dims(lr_image, axis=0)  # Add batch dimension

    # Predict the high-resolution image
    sr_image = model.predict(lr_image)[0]
    sr_image = (sr_image * 255).astype(np.uint8)  # Convert to uint8

    # Save the enhanced image
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
    output_file = os.path.join(output_path, f"enhanced_{img_name}")
    cv2.imwrite(output_file, cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))
    print(f"Saved enhanced image: {output_file}")

# Define input and output paths
lr_image_path = r"C:\Users\hp\Desktop\datasett\dataset\Raw Data\low_res\13.png"  # Single image file
output_path = r"C:\Users\hp\Desktop\datasett\dataset\Enhanced"  # Output directory

# Enhance and save the image
enhance_image(lr_image_path, output_path)

