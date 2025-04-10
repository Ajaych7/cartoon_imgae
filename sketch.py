import cv2

def pencil_sketch_cartoon(img_path):
    img = cv2.imread(img_path)
    
    # Pencil sketch (grayscale + inverted blur)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - gray_blur, scale=256)
    
    # Color reduction (posterization)
    reduced_color = cv2.stylization(img, sigma_s=60, sigma_r=0.45)
    
    # Combine sketch and color
    cartoon = cv2.bitwise_and(reduced_color, reduced_color, mask=sketch)
    cv2.imwrite("pencil_cartoon.jpg", cartoon)
    return cartoon

pencil_sketch_cartoon("input.jpg")
