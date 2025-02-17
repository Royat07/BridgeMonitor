import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import cv2

def show_img (title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def build_unet (input_shape):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 =layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def load_data(img_dir, mask_dir):
    images = []
    masks = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(mask_dir, mask_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        images.append(img)
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (256, 256))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255
        masks.append(mask)
    return np.array(images), np.array(masks)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--img_dir', type=str, required=True, help='Images directory.')
arg_parser.add_argument('-m', '--mask_dir', type=str, required=True, help='Masks directory.')
args = arg_parser.parse_args()
img_dir = args.img_dir
mask_dir = args.mask_dir

images, masks = load_data(img_dir, mask_dir)
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
input_shape = (256, 256, 3)
if os.path.exists("crack_detector_model.keras"):
    model = tf.keras.models.load_model("crack_detector_model.keras")
else:
    model = build_unet(input_shape)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_val, y_val))
    model.save("crack_detector_model.keras")
model.summary()

idx = np.random.randint(0, len(X_val))
test_img = X_val[idx]
true_mask = y_val[idx]

test_img_input = np.expand_dims(test_img, axis=0)
pred_mask = model.predict(test_img_input)[0]

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(test_img)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(true_mask, cmap='gray')
plt.title('True Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()