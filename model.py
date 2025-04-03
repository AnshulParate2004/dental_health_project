from PIL import Image
import numpy as np

def analyze_teeth(image_path):
    try:
        # Open and process the image
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Standard size for many models
        img_array = np.array(img) / 255.0  # Normalize
        
        # Simulated analysis (replace with real ML model inference)
        avg_color = np.mean(img_array)
        if avg_color > 0.8:
            recommendation = "Teeth look healthy! Maintain regular brushing and flossing."
        elif avg_color > 0.5:
            recommendation = "Slight discoloration detected. Consider a dental checkup."
        else:
            recommendation = "Potential issues detected. Visit a dentist soon."
        
        return recommendation
    except Exception as e:
        return f"Error analyzing image: {str(e)}"