def watercolor_cartoon(img_path):
    img = cv2.imread(img_path)
    cartoon = cv2.stylization(img, sigma_s=100, sigma_r=0.8)  # Soft edges
    cv2.imwrite("watercolor_cartoon.jpg", cartoon)
    return cartoon

watercolor_cartoon("input.jpg")
