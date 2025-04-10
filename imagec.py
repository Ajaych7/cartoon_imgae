import cv2
import numpy as np

def cartoonize_image(img_path, ksize=5, sketch_mode=False):
    # Read the image
    img = cv2.imread(img_path)
    
    # Apply bilateral filter for smoothing while preserving edges
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Convert to grayscale and apply adaptive threshold for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, ksize)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, blockSize=9, C=2
    )
    
    # Combine smoothed image with edges
    if sketch_mode:
        # Black-and-white sketch effect
        cartoon = cv2.bitwise_and(gray, gray, mask=edges)
        return cv2.cvtColor(cartoon, cv2.COLOR_GRAY2BGR)
    else:
        # Color cartoon effect
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(smoothed, edges_color)
        return cartoon

# Example usage
input_image = "input.jpg"
output_cartoon = cartoonize_image(input_image, sketch_mode=False)

# Save the result
cv2.imwrite("cartoon_output.jpg", output_cartoon)
print("Cartoon image saved as 'cartoon_output.jpg'")
