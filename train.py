#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import keras
import keras.backend as K
import re
import cv2
import numpy as np
np.set_printoptions(threshold='nan')


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def get_train_test_dataset():
    if os.path.exists('./data/train.npz'):
        dataset = np.load('./data/train.npz')
        print('{} already exits.'.format('./data/train.npz'))
        return (dataset['x'], dataset['y'])

    x = list_pictures('./test_dataset', ext='png')
    y = [item[:-4] + '_posmap.jpg' for item in x]
    filted_x = []
    filted_y = []
    for ix, iy in zip(x, y):
        if os.path.exists(ix) and os.path.exists(iy):
            filted_x.append(ix)
            filted_y.append(iy)
        else:
            print('{} or {} not exits.'.format(ix, iy))
    x = [cv2.imread(item) for item in filted_x]
    y = [cv2.imread(item) for item in filted_y]
    x = np.array(x)
    y = np.array(y)
    if not os.path.exists('./data'):
        os.makedirs('./data')
    np.savez('./data/train.npz', x=x, y=y)
    return (x, y)


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


def preprocess_input(x, y=None):
    x = x.astype(np.float32)
    x = keras.applications.xception.preprocess_input(x)
    if y is not None:
        y = y.astype(np.float32)
        y /= 256.0
    return (x, y)


loss_mask = cv2.imread('./data/uv-data/uv_weight_mask.png')
face_mask = cv2.imread('./data/uv-data/uv_face_mask.png')
loss_mask = np.where(face_mask > 0, loss_mask, face_mask)
loss_mask = loss_mask.astype(np.float32)
loss_mask /= 16.0


def mean_squared_error_with_mask(y_true, y_pred):
    mask = K.constant(loss_mask)
    return K.mean(K.mean(K.square(y_pred - y_true) * mask, axis=-1), axis=-1)


def lr_adjustor(epoch):
    base_lr = 0.001
    if epoch < 100:
        return base_lr
    base_lr *= .1
    if epoch < 150:
        return base_lr
    base_lr *= .1
    return base_lr


def train():
    (x, y) = get_train_test_dataset()
    # x = np.concatenate([x for i in range(20)])
    # y = np.concatenate([y for i in range(20)])
    print('x shape -> {}, y shape -> {}.'.format(x.shape, y.shape))
    (x, y) = preprocess_input(x, y)

    model = get_regress_model()
    model.summary()
    model.load_weights('./weights.100-0.0137.hdf5')
    # keras.utils.plot_model(model, show_shapes=True)
    opti = keras.optimizers.Adam(lr=0.001)
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_adjustor),
        keras.callbacks.CSVLogger('train.log'),
        keras.callbacks.ModelCheckpoint(
            './weights/weights.{epoch:02d}-{loss:.4f}.hdf5',
            monitor='loss',
            save_best_only=True,
            period=10)]
    model.compile(opti, loss=mean_squared_error_with_mask)
    model.fit(x, y, batch_size=16, epochs=200, callbacks=callbacks)


def test():
    (x, y) = get_train_test_dataset()
    # x = np.concatenate([x for i in range(20)])
    # y = np.concatenate([y for i in range(20)])
    print('x shape -> {}, y shape -> {}.'.format(x.shape, y.shape))
    (x, y) = preprocess_input(x, y)

    model = get_regress_model()
    model.summary()
    # model.load_weights('./weights.100-0.0137.hdf5')
    model.load_weights('./Data/net-data/weights.190-0.0010.hdf5')

    if not os.path.exists('./result'):
        os.makedirs('./result')
    y = model.predict(x)
    for index, i in enumerate(y):
        i *= 255
        i = i.astype(np.uint8)
        savename = os.path.join('./result', str(index) + '.png')
        cv2.imwrite(savename, i)


if __name__ == "__main__":
    # train()
    test()
