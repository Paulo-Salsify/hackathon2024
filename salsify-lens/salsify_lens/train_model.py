import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Image data generator for data augmentation
#datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# Update your ImageDataGenerator for training with more augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
#train_dir = os.path.join(os.getcwd(), 'train', 'magic_mouse')
train_dir = os.path.join(os.getcwd(), 'train')
image_size = (224, 224)  # 100 or (224, 224) for more complex models
image_channels = 3  # 1 for grayscale, 3 for RGB
model_filename= 'object_detection_model2.h5'
folders_to_process = ['magic_mouse3'] #['magic_mouse', 'magic_mouse2']


def pad_image(image, target_size):
    """Pad the image to make it square while keeping the aspect ratio."""
    width, height = image.size
    new_image = Image.new("RGB", target_size)  # Create a new black image
    # Calculate padding
    x_offset = (target_size[0] - width) // 2
    y_offset = (target_size[1] - height) // 2
    # Paste the original image onto the black image
    new_image.paste(image, (x_offset, y_offset))
    return new_image

def process_images(source_folder, target_folder, size=image_size, grayscale=False):
    """
    Reads all images from source_folder, resizes them to specified size,
    and pads them to keep aspect ratio, then saves them to target_folder.
    Optionally converts images to grayscale.
    """
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Loop through all files in the source folder
    file_number = 1
    for filename in os.listdir(source_folder):
        # Only process image files (add other extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Open image
                img_path = os.path.join(source_folder, filename)
                img = Image.open(img_path)

                # Resize while keeping aspect ratio
                img.thumbnail((size[0], size[1]), Image.ANTIALIAS)

                # Pad the image to make it square
                img_padded = pad_image(img, size)

                # Convert to grayscale if specified
                if grayscale:
                    img_padded = img_padded.convert('L')  # Convert to grayscale

                # Save the processed image
                target_path = os.path.join(target_folder, f"file_{file_number}.jpg")
                img_padded.save(target_path, format='JPEG')
                
                print(f"Processed and saved: {filename}")
                file_number += 1
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Convert raw images to grayscale and resize
for folder in folders_to_process:
    process_images(
        os.path.join(os.getcwd(), 'raw_images', folder),
        os.path.join(os.getcwd(), 'train', folder),
    )

train_data = datagen.flow_from_directory(train_dir, 
                                         target_size=image_size,
                                         batch_size=32,
                                         class_mode='binary', # Use 'binary' for two classes; use 'categorical' if you have more than two classes.
                                         subset='training')

print("Validating data...")
val_data = datagen.flow_from_directory(train_dir,
                                       target_size=image_size,
                                       batch_size=32,
                                       class_mode='binary', # Use 'binary' for two classes; use 'categorical' if you have more than two classes.
                                       subset='validation')

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*image_size, image_channels)), #(*image_size, 3image_channels) 1 = grayscale 3 = colour
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile and train the model
print("Compile model...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model fit...")
model.fit(train_data, epochs=10, validation_data=val_data)

# Save the trained model
print("Saving model...")
model.save(os.path.join(os.getcwd(), 'models', 'keras', model_filename))