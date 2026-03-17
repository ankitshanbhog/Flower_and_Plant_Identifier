import tensorflow as tf
from tensorflow.keras import layers, models
import os
import pickle

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 
DATA_DIR = 'dataset'

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'val'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
os.makedirs('models', exist_ok=True)
with open('models/classes.pkl', 'wb') as f:
    pickle.dump(class_names, f)


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    data_augmentation, # Helps the model learn better
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n🚀 Training on {len(class_names)} classes with Data Augmentation...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save('models/plant_classifier_v1.h5')
print("\n✅ Done! Run your frontend next.")