import os
import json
import numpy as np
import cv2
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError


model_filename = 'annotations_coca_cola_model.h5'
train_folder = 'coca_cola'
train_label = 'coca-cola'


def load_annotation(image_file, annotations_path):
    # Construct the corresponding JSON filename without relying on `imagePath`
    base_filename = os.path.splitext(image_file)[0]
    annotation_file = os.path.join(annotations_path, f"{base_filename}.json")
    
    # Check if the JSON annotation file exists
    if not os.path.exists(annotation_file):
        print(f"[DEBUG] No annotation found for: {image_file}")
        return []  # Return an empty list if no annotation file is found
    
    # Load and parse JSON annotation
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] Failed to parse JSON for {annotation_file}: {e}")
        return []
    
    # Extract bounding boxes from annotation, ignoring `imagePath`
    bounding_boxes = []
    for shape in data.get('shapes', []):
        if shape.get('label') == train_label:  # Make sure this label matches your LabelMe annotation
            points = shape.get('points', [])
            if points:
                x_min = int(min(point[0] for point in points))
                y_min = int(min(point[1] for point in points))
                x_max = int(max(point[0] for point in points))
                y_max = int(max(point[1] for point in points))
                bounding_boxes.append((x_min, y_min, x_max, y_max))
    
    # Debug: Print bounding boxes for the image
    if bounding_boxes:
        print(f"[DEBUG] Bounding boxes for {image_file}: {bounding_boxes}")
    else:
        print(f"[DEBUG] No bounding boxes found in {annotation_file}")
    
    return bounding_boxes


def load_data(images_path, annotations_path):
    images = []
    labels = []
    
    for image_file in os.listdir(images_path):
        if image_file.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(images_path, image_file)
            annotation = load_annotation(image_file, annotations_path)
            
            if annotation:  # Only add if there’s a valid annotation
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(annotation)
            else:
                print(f"[DEBUG] Skipping image {image_file} due to missing annotation.")

    if not images or not labels:
        raise ValueError("No valid image-annotation pairs found. Check your dataset and annotations.")
    
    print(f"[DEBUG] Loaded {len(images)} images and {len(labels)} annotations.")
    return np.array(images), labels

def convert_labels_to_format(labels, input_size=(224, 224)):
    formatted_labels = []
    for bboxes in labels:
        formatted = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            formatted.append([
                x_min / input_size[0],
                y_min / input_size[1],
                x_max / input_size[0],
                y_max / input_size[1]
            ])
        formatted_labels.append(formatted)
    return np.array(formatted_labels, dtype='float32')  # Convert to NumPy array here

def create_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4, activation='linear')(x)  # Output layer for 4 coordinates
    model = Model(inputs=base_model.input, outputs=x)
    return model

images_path = os.path.join(os.getcwd(), 'raw_images', train_folder, 'images')
annotations_path = os.path.join(os.getcwd(), 'raw_images', train_folder, 'annotations')

# Attempt to load data with debugging info
images, labels = load_data(images_path, annotations_path)
formatted_labels = convert_labels_to_format(labels)

# Confirm that formatted_labels is now a NumPy array
print(f"[DEBUG] Shape of formatted_labels: {formatted_labels.shape}")

# Create and compile the model with a regression loss
model = create_model()
#model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['mae'])

early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    os.path.join(os.getcwd(), 'models', 'keras', model_filename), save_best_only=True
)

# Confirm shapes of images and labels
print(f"[DEBUG] Shape of images: {images.shape}")
print(f"[DEBUG] Shape of formatted_labels: {formatted_labels.shape}")

print("[DEBUG] Starting model training...")
model.fit(images, formatted_labels, epochs=20, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
