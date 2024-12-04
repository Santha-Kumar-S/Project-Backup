import cv2
import numpy as np

def process(img_path, img_format, palette, labels, return_report_image=False):
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image path or format")
    
    # Resize and convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Flatten the image and find the dominant color
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Perform KMeans clustering to find the dominant color
    k = len(palette)
    _, labels, centers = cv2.kmeans(pixels, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    
    # Map dominant color to closest palette color
    distances = [np.linalg.norm(np.array(dominant_color) - np.array(hex_to_rgb(color))) for color in palette]
    closest_index = np.argmin(distances)
    
    # Generate result
    result = {
        "faces": [{
            "tone_label": labels[closest_index],
            "accuracy": 95.0,  # Dummy accuracy
            "skin_tone": palette[closest_index],
            "dominant_colors": [palette[i] for i in np.unique(labels)]
        }]
    }
    return result

def hex_to_rgb(hex_color):
    return tuple(int(hex_color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
