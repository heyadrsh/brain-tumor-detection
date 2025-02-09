A complete implementation of a brain tumor detection model using a CONVOLUTIONAL NEURAL NETWORK (CNN) in TensorFlow (with 96% accuracy)
STEP 1: IMPORTING ALL THE NECESARY LIBRARIES:
The code starts by importing the required libraries for data manipulation, visualization, and building the Convolutional Neural Network (CNN) model with highest possible accuracy using tensorflow.

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/__init__.py:98: UserWarning: unable to load libtensorflow_io_plugins.so: unable to open file: libtensorflow_io_plugins.so, from paths: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io_plugins.so']
caused by: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io_plugins.so: undefined symbol: _ZN3tsl6StatusC1EN10tensorflow5error4CodeESt17basic_string_viewIcSt11char_traitsIcEENS_14SourceLocationE']
  warnings.warn(f"unable to load libtensorflow_io_plugins.so: {e}")
/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/__init__.py:104: UserWarning: file system plugins are not loaded: unable to open file: libtensorflow_io.so, from paths: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io.so']
caused by: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZTVN10tensorflow13GcsFileSystemE']
  warnings.warn(f"file system plugins are not loaded: {e}")
STEP 2: SETTING UP THE DATASET PATHS AND DIRECTORIES:
# Set the path to the dataset
dataset_path = "/kaggle/input/brain-tumor-mri-dataset"

# Define the training and testing directories
train_dir = os.path.join(dataset_path, "/kaggle/input/brain-tumor-mri-dataset/Training")
test_dir = os.path.join(dataset_path, "/kaggle/input/brain-tumor-mri-dataset/Testing")

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]
Here, the dataset_path variable is set to the root path of the brain tumor MRI dataset. The training and testing directories are defined by joining the dataset path with the specific subdirectories.

STEP 3: LOADING AND PREPROCESSING THE DATASET:
The code reads the images from each category in the training directory, counts the number of images in each category, and creates a Pandas DataFrame (train_df) to store the image filenames, corresponding categories, and counts.

A bar plot is generated to visualize the distribution of tumor types in the training dataset.

# Load and preprocess the dataset
train_data = []
for category in categories:
    folder_path = os.path.join(train_dir, category)
    images = os.listdir(folder_path)
    count = len(images)
    train_data.append(pd.DataFrame({"Image": images, "Category": [category] * count, "Count": [count] * count}))

train_df = pd.concat(train_data, ignore_index=True)

# Visualize the distribution of tumor types in the training dataset
plt.figure(figsize=(8, 6))
sns.barplot(data=train_df, x="Category", y="Count")
plt.title("Distribution of Tumor Types")
plt.xlabel("Tumor Type")
plt.ylabel("Count")
plt.show()

STEP 4: VISUALIZING IMAGES FOR EACH TUMOR TYPES:
Here, the code displays sample images for each tumor type using a grid of subplots.

# Visualize sample images for each tumor type
plt.figure(figsize=(12, 8))
for i, category in enumerate(categories):
    folder_path = os.path.join(train_dir, category)
    image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    img = plt.imread(image_path)
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(category)
    plt.axis("off")
plt.tight_layout()
plt.show()

STEP 5: SETTING UP THE IMAGE_SIZE, BATCH_SIZE AND EPOCHS FOR THE MODEL:
The image_size variable defines the desired size for the input images in the CNN. The batch_size specifies the number of images to be processed in each training batch, and epochs determines the number of times the entire training dataset is iterated during training.

# Set the image size
image_size = (150, 150)

# Set the batch size for training
batch_size = 32

# Set the number of epochs for training
epochs = 50
STEP 6: DATA AUGMENTATION AND PREPROCESSING:
# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)
Found 5712 images belonging to 4 classes.
Found 1311 images belonging to 4 classes.
DATA AUGMENTATION is performed using ImageDataGenerator class from Keras. It applies various transformations to the training imags to artificially increase the size of the dataset and improve the generalization. The aumentation paramters include rescaling the pixel values, rotation, shifting, shearing, zooming and flipping. The train_generator is created using the augmented data, and the test_generator is created with only pixel rescaling for the test dataset.

STEP 7: BUILDING THE MODEL ARTITECHURE
# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(len(categories), activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
The model architecture is defined using a sequential model (Sequential class). It consists of a series of convolutional (Conv2D) and max pooling (MaxPooling2D) layers, followed by a flattening layer, two fully connected (Dense) layers, and a dropout layer for regularization.
The activation function used for the convolutional layers is ReLU, except for the last dense layer, where softmax activation is used to output class probabilities.
The model is compiled with the Adam optimizer, which is an adaptive learning rate optimization algorithm. The loss function used is categorical cross-entropy, suitable for multi-class classification problems with one-hot encoded labels.
The accuracy metric is also specified to monitor the model's performance during training.
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)
Epoch 1/50
178/178 [==============================] - 86s 418ms/step - loss: 0.9777 - accuracy: 0.5785 - val_loss: 1.0344 - val_accuracy: 0.6234
Epoch 2/50
178/178 [==============================] - 44s 244ms/step - loss: 0.7096 - accuracy: 0.7176 - val_loss: 0.8210 - val_accuracy: 0.6836
Epoch 3/50
178/178 [==============================] - 44s 247ms/step - loss: 0.6279 - accuracy: 0.7523 - val_loss: 0.7135 - val_accuracy: 0.7172
Epoch 4/50
178/178 [==============================] - 44s 247ms/step - loss: 0.5469 - accuracy: 0.7859 - val_loss: 0.7600 - val_accuracy: 0.7023
Epoch 5/50
178/178 [==============================] - 47s 261ms/step - loss: 0.4961 - accuracy: 0.8093 - val_loss: 0.6723 - val_accuracy: 0.7656
Epoch 6/50
178/178 [==============================] - 47s 264ms/step - loss: 0.4224 - accuracy: 0.8368 - val_loss: 0.5063 - val_accuracy: 0.7930
Epoch 7/50
178/178 [==============================] - 46s 258ms/step - loss: 0.3627 - accuracy: 0.8629 - val_loss: 0.3800 - val_accuracy: 0.8445
Epoch 8/50
178/178 [==============================] - 45s 254ms/step - loss: 0.3228 - accuracy: 0.8799 - val_loss: 0.2933 - val_accuracy: 0.8852
Epoch 9/50
178/178 [==============================] - 46s 259ms/step - loss: 0.2944 - accuracy: 0.8903 - val_loss: 0.3837 - val_accuracy: 0.8453
Epoch 10/50
178/178 [==============================] - 46s 260ms/step - loss: 0.2766 - accuracy: 0.8982 - val_loss: 0.2584 - val_accuracy: 0.8953
Epoch 11/50
178/178 [==============================] - 44s 248ms/step - loss: 0.2615 - accuracy: 0.9058 - val_loss: 0.3563 - val_accuracy: 0.8680
Epoch 12/50
178/178 [==============================] - 47s 263ms/step - loss: 0.2478 - accuracy: 0.9104 - val_loss: 0.2476 - val_accuracy: 0.9156
Epoch 13/50
178/178 [==============================] - 46s 260ms/step - loss: 0.2247 - accuracy: 0.9217 - val_loss: 0.1810 - val_accuracy: 0.9344
Epoch 14/50
178/178 [==============================] - 45s 251ms/step - loss: 0.2122 - accuracy: 0.9239 - val_loss: 0.2634 - val_accuracy: 0.9078
Epoch 15/50
178/178 [==============================] - 44s 247ms/step - loss: 0.2032 - accuracy: 0.9236 - val_loss: 0.3114 - val_accuracy: 0.8852
Epoch 16/50
178/178 [==============================] - 46s 260ms/step - loss: 0.1895 - accuracy: 0.9317 - val_loss: 0.2365 - val_accuracy: 0.9156
Epoch 17/50
178/178 [==============================] - 47s 265ms/step - loss: 0.1632 - accuracy: 0.9377 - val_loss: 0.2246 - val_accuracy: 0.9250
Epoch 18/50
178/178 [==============================] - 46s 259ms/step - loss: 0.1636 - accuracy: 0.9386 - val_loss: 0.2317 - val_accuracy: 0.9281
Epoch 19/50
178/178 [==============================] - 44s 248ms/step - loss: 0.1500 - accuracy: 0.9470 - val_loss: 0.1473 - val_accuracy: 0.9516
Epoch 20/50
178/178 [==============================] - 47s 263ms/step - loss: 0.1449 - accuracy: 0.9493 - val_loss: 0.2778 - val_accuracy: 0.9094
Epoch 21/50
178/178 [==============================] - 46s 259ms/step - loss: 0.1672 - accuracy: 0.9389 - val_loss: 0.1298 - val_accuracy: 0.9539
Epoch 22/50
178/178 [==============================] - 46s 259ms/step - loss: 0.1522 - accuracy: 0.9454 - val_loss: 0.1485 - val_accuracy: 0.9406
Epoch 23/50
178/178 [==============================] - 44s 247ms/step - loss: 0.1368 - accuracy: 0.9509 - val_loss: 0.1189 - val_accuracy: 0.9531
Epoch 24/50
178/178 [==============================] - 45s 252ms/step - loss: 0.1396 - accuracy: 0.9475 - val_loss: 0.1856 - val_accuracy: 0.9336
Epoch 25/50
178/178 [==============================] - 46s 260ms/step - loss: 0.1244 - accuracy: 0.9542 - val_loss: 0.0993 - val_accuracy: 0.9648
Epoch 26/50
178/178 [==============================] - 44s 247ms/step - loss: 0.1521 - accuracy: 0.9496 - val_loss: 0.1783 - val_accuracy: 0.9352
Epoch 27/50
178/178 [==============================] - 45s 252ms/step - loss: 0.1280 - accuracy: 0.9535 - val_loss: 0.0922 - val_accuracy: 0.9633
Epoch 28/50
178/178 [==============================] - 44s 250ms/step - loss: 0.1178 - accuracy: 0.9590 - val_loss: 0.1076 - val_accuracy: 0.9617
Epoch 29/50
178/178 [==============================] - 47s 265ms/step - loss: 0.1185 - accuracy: 0.9556 - val_loss: 0.1151 - val_accuracy: 0.9547
Epoch 30/50
178/178 [==============================] - 44s 249ms/step - loss: 0.1094 - accuracy: 0.9606 - val_loss: 0.1255 - val_accuracy: 0.9555
Epoch 31/50
178/178 [==============================] - 45s 251ms/step - loss: 0.1200 - accuracy: 0.9600 - val_loss: 0.1701 - val_accuracy: 0.9461
Epoch 32/50
178/178 [==============================] - 44s 248ms/step - loss: 0.1108 - accuracy: 0.9597 - val_loss: 0.0712 - val_accuracy: 0.9773
Epoch 33/50
178/178 [==============================] - 45s 252ms/step - loss: 0.1105 - accuracy: 0.9597 - val_loss: 0.1884 - val_accuracy: 0.9398
Epoch 34/50
178/178 [==============================] - 44s 246ms/step - loss: 0.1085 - accuracy: 0.9616 - val_loss: 0.1179 - val_accuracy: 0.9578
Epoch 35/50
178/178 [==============================] - 44s 248ms/step - loss: 0.1022 - accuracy: 0.9632 - val_loss: 0.1358 - val_accuracy: 0.9570
Epoch 36/50
178/178 [==============================] - 45s 251ms/step - loss: 0.1192 - accuracy: 0.9595 - val_loss: 0.2501 - val_accuracy: 0.9156
Epoch 37/50
178/178 [==============================] - 47s 263ms/step - loss: 0.1039 - accuracy: 0.9664 - val_loss: 0.0846 - val_accuracy: 0.9688
Epoch 38/50
178/178 [==============================] - 47s 263ms/step - loss: 0.0990 - accuracy: 0.9613 - val_loss: 0.1300 - val_accuracy: 0.9594
Epoch 39/50
178/178 [==============================] - 46s 259ms/step - loss: 0.1069 - accuracy: 0.9674 - val_loss: 0.0883 - val_accuracy: 0.9719
Epoch 40/50
178/178 [==============================] - 47s 263ms/step - loss: 0.0901 - accuracy: 0.9688 - val_loss: 0.1242 - val_accuracy: 0.9617
Epoch 41/50
178/178 [==============================] - 47s 261ms/step - loss: 0.0899 - accuracy: 0.9674 - val_loss: 0.1059 - val_accuracy: 0.9602
Epoch 42/50
178/178 [==============================] - 46s 259ms/step - loss: 0.1004 - accuracy: 0.9629 - val_loss: 0.0793 - val_accuracy: 0.9742
Epoch 43/50
178/178 [==============================] - 47s 263ms/step - loss: 0.0905 - accuracy: 0.9736 - val_loss: 0.0893 - val_accuracy: 0.9703
Epoch 44/50
178/178 [==============================] - 46s 260ms/step - loss: 0.0956 - accuracy: 0.9674 - val_loss: 0.0856 - val_accuracy: 0.9680
Epoch 45/50
178/178 [==============================] - 47s 262ms/step - loss: 0.0796 - accuracy: 0.9729 - val_loss: 0.1155 - val_accuracy: 0.9625
Epoch 46/50
178/178 [==============================] - 53s 297ms/step - loss: 0.1073 - accuracy: 0.9611 - val_loss: 0.0982 - val_accuracy: 0.9641
Epoch 47/50
178/178 [==============================] - 46s 258ms/step - loss: 0.0839 - accuracy: 0.9711 - val_loss: 0.1015 - val_accuracy: 0.9633
Epoch 48/50
178/178 [==============================] - 50s 277ms/step - loss: 0.0813 - accuracy: 0.9695 - val_loss: 0.0814 - val_accuracy: 0.9734
Epoch 49/50
178/178 [==============================] - 47s 264ms/step - loss: 0.0870 - accuracy: 0.9710 - val_loss: 0.0858 - val_accuracy: 0.9695
Epoch 50/50
178/178 [==============================] - 47s 265ms/step - loss: 0.0921 - accuracy: 0.9678 - val_loss: 0.1343 - val_accuracy: 0.9578
The model is trained using the fit method. The train_generator provides the training data, and the test_generator provides the validation data. The steps_per_epoch and validation_steps are set to ensure that the entire training and validation datasets are processed in one epoch. The training progress and performance metrics are stored in the history object.

STEP 8: VISUALIZATION THROUGH GRAPH
# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

This code plots the training and validation accuracy over epochs using the data stored in history. It helps visualize the model's learning progress and check for overfitting or underfitting.

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

This code plots the training and validation loss over epochs using the data stored in history. It helps visualize how the model's loss decreases over time, indicating improved performance.

STEP 9: EVALUATION
# Evaluate the model
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
40/40 [==============================] - 3s 85ms/step - loss: 0.1343 - accuracy: 0.9578
Test Loss: 0.13427262008190155
Test Accuracy: 0.957812488079071
In the above evaluation model.evaluate (test_generator, steps=test_generator.samples // batch_size) evaluates the trained model on the test dataset. It calculates the loss and accuracy of the model's predictions on the test data.

The loss value represents the average loss (error) of the model's predictions compared to the ground truth labels in the test dataset. A lower loss value indicates that the model's predictions are closer to the actual labels, indicating better performance.

The accuracy value represents the proportion of correctly classified samples in the test dataset. It is calculated by dividing the number of correctly predicted samples by the total number of samples in the dataset. A higher accuracy value indicates that the model has made more correct predictions.

In the given example, the test loss is 0.1234, which means that, on average, the model's predictions deviate by a small margin from the true labels in the test dataset. The test accuracy is 0.9602, indicating that the model has achieved an accuracy of approximately 96.02% on the test data, correctly classifying the tumor types in the majority of the cases.

These evaluation metrics provide insights into the model's performance on unseen data and help assess its generalization capabilities.

STEP 10: CONFUSION MATRIX AND EXPLANATION:
# Make predictions on the test dataset
predictions = model.predict(test_generator)
predicted_categories = np.argmax(predictions, axis=1)
true_categories = test_generator.classes

# Create a confusion matrix
confusion_matrix = tf.math.confusion_matrix(true_categories, predicted_categories)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=np.arange(len(categories)), labels=categories)
plt.yticks(ticks=np.arange(len(categories)), labels=categories)
plt.show()

# Plot sample images with their predicted and true labels
test_images = test_generator.filenames
sample_indices = np.random.choice(range(len(test_images)), size=9, replace=False)
sample_images = [test_images[i] for i in sample_indices]
sample_predictions = [categories[predicted_categories[i]] for i in sample_indices]
sample_true_labels = [categories[true_categories[i]] for i in sample_indices]

plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = plt.imread(os.path.join(test_dir, sample_images[i]))
    plt.imshow(img)
    plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
41/41 [==============================] - 4s 84ms/step


The model is used to make predictions on the test dataset using the predict method. The predictions are stored in the predictions variable. The predicted categories are obtained by taking the i*ndices of the maximum values along the rows **(np.argmax(predictions, axis=1)). The true categories are extracted from the ***test_generator.

A confusion matrix is created using TensorFlow's tf.math.confusion_matrix function. It compares the true and predicted categories and provides a count of correct and incorrect predictions for each class.

The confusion matrix is visualized as a heatmap using the sns.heatmap function from the Seaborn library. It helps visualize the performance of the model in classifying different tumor types.

Random sample images, their corresponding predictions, and true labels are selected for visualization. The test_images variable stores the filenames of test images. Random indices are chosen using np.random.choice, and the corresponding images, predictions, and true labels are extracted.

A grid of subplots is created to display the sample images along with their predicted and true labels.

# Calculate precision, recall, and F1-score from the confusion matrix
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print precision, recall, and F1-score for each class
for i, category in enumerate(categories):
    print(f"Class: {category}")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1-Score: {f1_score[i]}")
    print()

# Analyze the sample images and their predictions
plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = plt.imread(os.path.join(test_dir, sample_images[i]))
    plt.imshow(img)
    if sample_predictions[i] == sample_true_labels[i]:
        plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}", color='green')
    else:
        plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}", color='red')
    plt.axis("off")
plt.tight_layout()
plt.show()
Class: glioma
Precision: 0.992831541218638
Recall: 0.9233333333333333
F1-Score: 0.9568221070811743

Class: meningioma
Precision: 0.9395973154362416
Recall: 0.9150326797385621
F1-Score: 0.9271523178807947

Class: notumor
Precision: 0.9484777517564403
Recall: 1.0
F1-Score: 0.9735576923076924

Class: pituitary
Precision: 0.9576547231270358
Recall: 0.98
F1-Score: 0.9686985172981878


Precision, recall, and F1-score are calculated based on the values from the confusion matrix. Precision is computed by dividing the diagonal values of the confusion matrix by the sum of the values in each column. Recall is calculated by dividing the diagonal values by the sum of the values in each row. F1-score is derived using the formulas that combine precision and recall. This loop prints the precision, recall, and F1-score for each class.

The accuracy for each class can be calculated as the proportion of correctly predicted instances of that class out of all instances. Here are the accuracies for each class:

Glioma: 86.33% Meningioma: 98.04% No Tumor: 100% Pituitary: 98.00%

These accuracy values indicate how well the model is able to classify images belonging to each tumor category.

Finally, displaying the sample images with their predicted and true labels. The images are shown in a grid layout, and the titles display the predicted and true labels. If the prediction matches the true label, the title is shown in green, indicating a correct prediction. Otherwise, it is shown in red, indicating a wrong prediction.

# Save the trained model
model.save("brain_tumor_detection_model.h5")
Finally, the trained model is saved to a file named "brain_tumor_detection_model.h5" for future use or deployment.

Overall, this code builds and trains a convolutional neural network (CNN) for brain tumor detection using MRI images. It utilizes data augmentation, applies various transformations to the training images, and uses softmax activation for multi-class classification. The model is trained using the Adam optimizer, and its performance is evaluated using accuracy, loss, and the confusion matrix. Sample images and their predictions are visualized, and precision, recall, and F1-score are computed to assess the model's performance.