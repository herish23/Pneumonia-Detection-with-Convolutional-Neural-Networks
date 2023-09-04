import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Dataset path
dataset_path = r"C:\Users\winni\Downloads\Compressed\archive_2\chest_xray"

# Lists to store data and labels
data = []
labels = []

# Define classes
types = ['NORMAL', 'PNEUMONIA']

# Load and preprocess training data
for label in types:
    full_path = os.path.join(dataset_path, 'train', label)
    for image_name in os.listdir(full_path):
        image_path = os.path.join(full_path, image_name)  # Correct the path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (128, 128))  # Resize the image
        data.append(image)
        labels.append(label)

# Load and preprocess testing data
testing_data = []
testing_labels = []

for label in types:
    full_path = os.path.join(dataset_path, 'test', label)
    for image_name in os.listdir(full_path):
        image_path = os.path.join(full_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (128, 128))  # Resize the image
        testing_data.append(image)
        testing_labels.append(label)

print("Data processed")
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
testing_labels_encoded = label_encoder.transform(testing_labels)

# Convert data to NumPy arrays and normalize
X_train = np.array(data) / 255.0
X_test = np.array(testing_data) / 255.0

# Reshape data to have a batch dimension
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Target labels
y_train = labels_encoded
y_test = testing_labels_encoded

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
y_pred = model.predict(X_test)
# binary labels (0 for 'NORMAL', 1 for 'PNEUMONIA')
y_pred_labels = (y_pred > 0.5).astype(int)
pneumonia_indices = np.where(y_pred_labels == 1)[0]
# Display 10 pneumonia images
num_images_to_display = 10
plt.figure(figsize=(15, 8))
for i, index in enumerate(pneumonia_indices[:num_images_to_display], 1):
    plt.subplot(2, 5, i)
    plt.imshow(X_test[index].reshape(128, 128), cmap='gray')
    plt.title('PNEUMONIA')
    plt.axis('off')

plt.tight_layout()
plt.show()



