import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

dataset_path = "dataset/train"

classes = os.listdir(dataset_path)
print("Total Classes:", len(classes))

# Display sample images
plt.figure(figsize=(12, 8))

for i, class_name in enumerate(classes[:6]):
    img_path = os.path.join(dataset_path, class_name, os.listdir(os.path.join(dataset_path, class_name))[0])
    img = load_img(img_path, target_size=(224, 224))
    
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.show()