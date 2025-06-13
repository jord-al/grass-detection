import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os

def detect_grass(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB for better visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    rgb_mask = np.zeros_like(mask)
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    blue_channel = image[:, :, 0]
    
    green_condition = (green_channel > red_channel) & (green_channel > blue_channel)
    intensity_condition = (green_channel > 40) & (green_channel < 200)
    rgb_mask[green_condition & intensity_condition] = 255
    
    # Combine HSV and RGB masks
    mask = cv2.bitwise_and(mask, rgb_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)  # Smaller kernel for more precise edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    min_size = 50
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    
    mask_cleaned = np.zeros_like(mask)
    for region in regions:
        if region.area >= min_size:
            if region.eccentricity < 0.9:
                mask_cleaned[labels == region.label] = 255
    
    # Calculate grass percentage
    total_pixels = image.shape[0] * image.shape[1]
    grass_pixels = np.sum(mask_cleaned > 0)
    grass_percentage = (grass_pixels / total_pixels) * 100
    
    # Create visualization
    result = image_rgb.copy()
    result[mask_cleaned > 0] = [0, 255, 0]  # Highlight grass in green
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save the result
    output_path = os.path.join('output', 'grass_detection_result.jpg')
    plt.imsave(output_path, result)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask_cleaned, cmap='gray')
    plt.title('Grass Mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(result)
    plt.title('Detected Grass')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'analysis_visualization.jpg'))
    plt.close()
    
    return grass_percentage, output_path

def main():
    os.makedirs('input', exist_ok=True)
    
    input_dir = 'input'
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("Please place an image in the 'input' directory")
        return
    
    image_path = os.path.join(input_dir, image_files[0])
    try:
        grass_percentage, output_path = detect_grass(image_path)
        print(f"\nAnalysis Results:")
        print(f"Grass Coverage: {grass_percentage:.2f}%")
        print(f"Processed image saved to: {output_path}")
        print(f"Analysis visualization saved to: output/analysis_visualization.jpg")
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 