from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
#from tensorflow import keras
import tensorflow as tf

#Step 2: Import the U-net model
from unet import *
img_size=(512,512)

n_class=5

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import scipy.misc as sc


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=True, n_class=n_class, save_to_dir=None, target_size=img_size, seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        class_mode=None,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        class_mode=None,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


def testGenerator(test_path, target_size=img_size, flag_multi_class=True, as_gray=True):
    files = sorted(os.listdir(test_path))
    num_image = len(files)
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, files[i]), as_gray=True)
        print(files[i])
        img = trans.resize(img, target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


# Step 4: Define function to save the test images
### draw imgs in labelVisualize and save results in saveResult
def saveResult(img_path, save_path, npyfile):
    files = os.listdir(img_path)

    for i, item in enumerate(npyfile):
        img = item
        for k in range(3):
            img[:, :, k] = img[:, :, k] / np.ptp(img[:, :, k])

        img[:, :, 1] = (img[:, :, 1] > 0.5).astype(
            int)  # This threshold of 0.05 can be changed to any number in range [0,1]
        img[:, :, 0] = (img[:, :, 0] > 0.5).astype(int)

        io.imsave(os.path.join(save_path, files[i]), img)


def SaveResultwImage(img_path, save_path, npyfile, target_size=img_size, flag_multi_class=True, num_class=2):
    files = os.listdir(img_path)

    for i, item in enumerate(npyfile):
        img = item
        img[img > 0.5] = 1
        img[img <= 0.5] = 0
        img[:, :, 2] = 0

        I = io.imread(os.path.join(img_path, files[i]), as_gray=True)
        I = trans.resize(I, target_size)
        img[:, :, 0] = np.true_divide((I + img[:, :, 0]), 2)
        img[:, :, 1] = np.true_divide((I + img[:, :, 1]), 2)
        img[:, :, 2] = np.true_divide((I + img[:, :, 2]), 2)
        io.imsave(os.path.join(save_path, files[i]), img)
    # Step 5: Define functions to evaluate the output


import sklearn.metrics as sm


def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """returns confusion matrix elements i.e TN, FP, FN, TP as floats
	See example code for helper function definitions
    """
    tn, fp, fn, tp = sm.confusion_matrix(groundtruth_list, predicted_list, labels=[0, 1]).ravel()
    tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

    return tn, fp, fn, tp


def get_prec_rec_IoU_accuracy(groundtruth_list, predicted_list):
    """returns precision, recall, IoU and accuracy metrics
	"""
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    IoU = tp / (tp + fp + fn)

    return prec, rec, IoU, accuracy


def get_f1_score(groundtruth_list, predicted_list):
    """Return f1 score covering edge cases"""

    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return f1_score


def get_validation_metrics(groundtruth, predicted):
    """Return all output metrics. Input is binary images"""

    u, v = np.shape(groundtruth)
    groundtruth_list = np.reshape(groundtruth, (u * v,))
    predicted_list = np.reshape(predicted, (u * v,))
    prec, rec, IoU, acc = get_prec_rec_IoU_accuracy(groundtruth_list, predicted_list)
    f1_score = get_f1_score(groundtruth_list, predicted_list)
    # print("Precision=",prec, "Recall=",rec, "IoU=",IoU, "acc=",acc, "F1=",f1_score)
    return prec, rec, IoU, acc, f1_score


def evalResult(gth_path, npyfile, target_size=img_size, flag_multi_class=False, num_class=3):
    files = sorted(os.listdir(gth_path))
    print(files)
    prec = 0
    rec = 0
    acc = 0
    IoU = 0
    f1_score = 0
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        gth = io.imread(os.path.join(gth_path, files[i]))
        gth = trans.resize(gth, target_size)
        img1 = np.array(((img - np.min(img)) / np.ptp(img)) > 0.1).astype(float)
        gth1 = np.array(((gth - np.min(gth)) / np.ptp(gth)) > 0.1).astype(float)
        p, r, I, a, f = get_validation_metrics(gth1, img1)
        prec = prec + p
        rec = rec + r
        acc = acc + a
        IoU = IoU + I
        f1_score = f1_score + f
    print("Precision=", prec / (i + 1), "Recall=", rec / (i + 1), "IoU=", IoU / (i + 1), "acc=", acc / (i + 1), "F1=",
          f1_score / (i + 1))


# Step 1: Call to image data generator in keras
data_gen_args = dict(rotation_range=0.3,
                     rescale=1. / 255,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.1,
                     zoom_range=[0.7, 1],
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')
PATH = 'train/'
if not os.path.exists(PATH + 'aug'):
    os.makedirs(PATH + 'aug')

if not os.path.exists(PATH + 'pred'):
    os.makedirs(PATH + 'pred')
data_gen = trainGenerator(3, PATH, 'train_val', 'mask_val', data_gen_args, save_to_dir=None)
for e in range(5):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in data_gen:
        print(np.max(x_batch))
        for i in range(0, 2):
            plt.subplot(330 + 1 + i)
            plt.imshow(y_batch[i], cmap=plt.get_cmap('gray'))

        plt.show()

        break

model = unet()
model.summary()

import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#Visualize on tensorboard (move this above)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_DB1_multi.hdf5', monitor='loss',verbose=0)
model.fit(data_gen,steps_per_epoch=20,epochs=100,verbose=1, callbacks=[model_checkpoint, tensorboard_callback])
