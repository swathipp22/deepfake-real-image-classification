#!/usr/bin/env python

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from PIL import Image
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import queue

base_path = '/data/swathi/project/archive/Dataset'

def plot_img(base_path, set_):
    dir_ = os.path.join(base_path, 'Train', set_)
    k = 0
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(set_ + ' Faces')
    for j in range(3):
        for i in range(3):
            img = load_img(os.path.join(dir_, os.listdir(os.path.join(dir_))[k]))
            ax[j, i].imshow(img)
            ax[j, i].set_title("")
            ax[j, i].axis('off')
            k += 1
    plt.suptitle(set_ + ' Faces')
    plt.show()

plot_img(base_path, 'Real')
plot_img(base_path, 'Fake')

ig = ImageDataGenerator(rescale=1./255.)
train_flow = ig.flow_from_directory(
    base_path + '/Train/',
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

ig1 = ImageDataGenerator(rescale=1./255.)
valid_flow = ig1.flow_from_directory(
    base_path + '/Validation/',
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

test_flow = ig.flow_from_directory(
    base_path + '/Test/',
    target_size=(128, 128),
    batch_size=1,
    shuffle=False,
    class_mode='categorical'
)

train_flow.class_indices

input_shape = (128, 128, 3)
batch_size = 64


def build_model():
    densenet = ResNet50(
        weights='/data/swathi/project/archive/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False,
        input_shape=input_shape
    )
    model = Sequential([densenet,
                        layers.GlobalAveragePooling2D(),
                        layers.Dense(512, activation='relu'),
                        layers.BatchNormalization(),
                        layers.Dense(2, activation='softmax')
                        ])
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


model = build_model()
model.summary()


class PredictionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(valid_flow[0][0])
        y_test = valid_flow[0][1]
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        print(y_pred_labels.shape)
        print(y_test_labels.shape)
        cfm = confusion_matrix(y_test_labels, y_pred_labels)
        print(cfm)
        print(y_pred[0], y_test[0])


train_steps = 140002 // batch_size
valid_steps = 10000 // batch_size

history = model.fit(train_flow,
                    epochs=2,
                    steps_per_epoch=train_steps,
                    validation_data=valid_flow,
                    validation_steps=valid_steps,
                    callbacks=[PredictionCallback()]
                    )

model.save('model.h5')
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'])

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'val'])
plt.show()

y_pred = model.predict(test_flow)
y_test = test_flow.classes
y_pred_labels = np.argmax(y_pred, axis=1)
confusion_matrix = confusion_matrix(y_test, y_pred_labels)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.show()

print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred_labels))
print("AP Score:", metrics.average_precision_score(y_test, y_pred_labels))
print()
print(metrics.classification_report(y_test, y_pred_labels))

_, accu = model.evaluate(test_flow)
print('Final Test Accuracy = {:.3f}'.format(accu*100))


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def predict_image(image_path, model_path='model.h5'):
    model = load_model(model_path)
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    if predicted_class == 1:
        result = 'Real'
    else:
        result = 'Fake'
    return result


if __name__ == "__main__":
    image_path = '/data/swathi/project/archive/Dataset/Train/Fake/fake_3456.jpg'
    prediction = predict_image(image_path)
    print("Prediction:", prediction)


image_queue = queue.Queue()
current_image = None

def browse_image():
    global current_image
    image_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                             filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if image_path:
        image_queue.put(image_path)  
        display_image(image_path)
        predict_button.config(state=tk.NORMAL)
        current_image = image_path


def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img


def predict():
    global current_image
    try:
        image_path = image_queue.get_nowait()  
        if image_path != current_image:
            messagebox.showerror("Error", "Please browse a new image for prediction!")
            image_queue.put(image_path)  
            return
        prediction = predict_image(image_path)
        result_label.config(text="Prediction: " + prediction)
        if image_queue.empty():
            predict_button.config(state=tk.DISABLED)
    except queue.Empty:
        messagebox.showerror("Error", "No image selected!")


root = tk.Tk()
root.title("Image Classifier")

browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

panel = tk.Label(root)
panel.pack()

predict_button = tk.Button(root, text="Predict", command=predict, state=tk.DISABLED)
predict_button.pack(pady=5)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()



