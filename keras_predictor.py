# -*- coding: utf-8 -*-
import os
import keras
import keras.backend as K
import re
import cv2
import numpy as np


def res_block(x, filters):
    # stage1
    shortcut = x
    shortcut = keras.layers.Conv2D(
        filters, (1, 1), strides=(2, 2), padding='same')(shortcut)
    x = keras.layers.Conv2D(
        filters / 2, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(
        filters / 2, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # stage2
    shortcut = x
    x = keras.layers.Conv2D(
        filters / 2, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(
        filters / 2, (4, 4), strides=(1, 1), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x


def get_regress_model():
    input = keras.layers.Input(shape=(256, 256, 3))
    x = keras.layers.Conv2D(
        16, (4, 4), strides=(1, 1), padding='same', activation='relu')(input)
    x = res_block(x, 32)
    x = res_block(x, 64)
    x = res_block(x, 128)
    x = res_block(x, 256)
    x = res_block(x, 512)

    x = keras.layers.Conv2DTranspose(512, (4, 4), padding='same', activation='relu')(x)

    x = keras.layers.Conv2DTranspose(
        256, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(256, (4, 4), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(256, (4, 4), padding='same', activation='relu')(x)

    x = keras.layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(128, (4, 4), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(128, (4, 4), padding='same', activation='relu')(x)

    x = keras.layers.Conv2DTranspose(
        64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(64, (4, 4), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(64, (4, 4), padding='same', activation='relu')(x)

    x = keras.layers.Conv2DTranspose(
        32, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(32, (4, 4), padding='same', activation='relu')(x)

    x = keras.layers.Conv2DTranspose(
        16, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(16, (4, 4), padding='same', activation='relu')(x)

    x = keras.layers.Conv2DTranspose(3, (4, 4), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(3, (4, 4), padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(3, (4, 4), padding='same')(x)

    model = keras.Model(input, x)
    return model


class PosPrediction():
    def __init__(self, resolution_inp = 256, resolution_op = 256): 
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp*1.1

        # network type
        self.network = get_regress_model()

    def restore(self, model_path):        
        # TODO(zzdxfei) model load
        self.network.load_weights(model_path)
 
    def predict(self, image):
        pos = self.network.predict(image[np.newaxis, :,:,:])
        pos = np.squeeze(pos)
        return pos*self.MaxPos

    def predict_batch(self, images):
        pos = self.network.predict(image)
        return pos*self.MaxPos

