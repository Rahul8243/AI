import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_size=(256,256,3), num_classes=3):

    inputs = layers.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(inputs)
    conv1 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2,2))(conv1)

    conv2 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool1)
    conv2 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2,2))(conv2)

    conv3 = layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool2)
    conv3 = layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2,2))(conv3)

    conv4 = layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool3)
    conv4 = layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2,2))(conv4)

    # Bottleneck
    conv5 = layers.Conv2D(1024,(3,3),activation='relu',padding='same')(pool4)
    conv5 = layers.Conv2D(1024,(3,3),activation='relu',padding='same')(conv5)

    # Decoder
    up6 = layers.UpSampling2D((2,2))(conv5)
    up6 = layers.Concatenate()([up6, conv4])
    conv6 = layers.Conv2D(512,(3,3),activation='relu',padding='same')(up6)
    conv6 = layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv6)

    up7 = layers.UpSampling2D((2,2))(conv6)
    up7 = layers.Concatenate()([up7, conv3])
    conv7 = layers.Conv2D(256,(3,3),activation='relu',padding='same')(up7)
    conv7 = layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv7)

    up8 = layers.UpSampling2D((2,2))(conv7)
    up8 = layers.Concatenate()([up8, conv2])
    conv8 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(up8)
    conv8 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv8)

    up9 = layers.UpSampling2D((2,2))(conv8)
    up9 = layers.Concatenate()([up9, conv1])
    conv9 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(up9)
    conv9 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv9)

    outputs = layers.Conv2D(num_classes,(1,1),activation='softmax')(conv9)

    model = models.Model(inputs, outputs)
    return model


model = unet_model(input_size=(256,256,3), num_classes=3)
model.summary()

img = Image.open(r"lab-5/Bennett_University_.jpg")  
original_width, original_height = img.size


img = img.resize((256,256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array[:,:,:3], axis=0)
img_array = img_array / 255.0


predictions = model.predict(img_array)

predictions = np.squeeze(predictions, axis=0)
predictions = np.argmax(predictions, axis=-1)
predictions = (predictions * 127).astype(np.uint8)

pred_img = Image.fromarray(predictions)
pred_img = pred_img.resize((original_width, original_height))


pred_img.save("predicted_image.jpg")
pred_img.show()