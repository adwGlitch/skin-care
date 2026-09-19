import os
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
SAMPLES_PER_CLASS = 25
RAW_DIR = os.path.join("ai", "dataset", "raw")
METADATA_PATH = os.path.join(RAW_DIR, "metadata.csv")

def generate_synthetic_image(output_path, class_name):
    skin_r = random.randint(210, 245)
    skin_g = random.randint(170, 210)
    skin_b = random.randint(150, 190)
    
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 0] = np.clip(np.random.normal(skin_r, 8, (224, 224)), 0, 255).astype(np.uint8)
    img_array[:, :, 1] = np.clip(np.random.normal(skin_g, 8, (224, 224)), 0, 255).astype(np.uint8)
    img_array[:, :, 2] = np.clip(np.random.normal(skin_b, 8, (224, 224)), 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    center_x = 112 + random.randint(-15, 15)
    center_y = 112 + random.randint(-15, 15)
    radius_x = random.randint(30, 55)
    radius_y = random.randint(30, 55)
    
    if class_name in ['mel', 'nv']:
        lesion_color = (random.randint(40, 90), random.randint(25, 55), random.randint(20, 50))
    elif class_name == 'vasc':
        lesion_color = (random.randint(160, 220), random.randint(20, 60), random.randint(30, 70))
    elif class_name == 'bcc':
        lesion_color = (random.randint(160, 200), random.randint(100, 140), random.randint(110, 150))
    elif class_name == 'akiec':
        lesion_color = (random.randint(180, 220), random.randint(130, 160), random.randint(100, 130))
    elif class_name == 'bkl':
        lesion_color = (random.randint(90, 130), random.randint(70, 100), random.randint(50, 80))
    else:
        lesion_color = (random.randint(120, 150), random.randint(80, 110), random.randint(60, 90))
        
    bbox = [center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y]
    draw.ellipse(bbox, fill=lesion_color)
    img.save(output_path, quality=95)

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    metadata_rows = []
    
    img_counter = 1
    for cls in CLASSES:
        for i in range(SAMPLES_PER_CLASS):
            image_id = f"ISIC_{img_counter:07d}"
            lesion_id = f"HAM_{img_counter:07d}"
            img_path = os.path.join(RAW_DIR, f"{image_id}.jpg")
            
            generate_synthetic_image(img_path, cls)
            
            metadata_rows.append({
                "lesion_id": lesion_id,
                "image_id": image_id,
                "dx": cls,
                "dx_type": "histo",
                "age": random.randint(20, 80),
                "sex": random.choice(["male", "female"]),
                "localization": random.choice(["back", "lower extremity", "trunk", "upper extremity", "abdomen", "face"])
            })
            img_counter += 1
            
    df = pd.DataFrame(metadata_rows)
    df.to_csv(METADATA_PATH, index=False)
    print(f"Successfully generated {len(metadata_rows)} sample images in {RAW_DIR}")
    print(f"Saved metadata to {METADATA_PATH}")

if __name__ == "__main__":
    main()
