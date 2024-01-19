import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize image data to [0, 1] 
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# Add a channel dimension
train_imgs = train_imgs.reshape((-1, 28, 28, 1))
test_imgs = test_imgs.reshape((-1, 28, 28, 1))

# Convert labels to categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build CNN Model
cnn_model = Sequential([
  Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(train_imgs, train_labels, epochs=10, batch_size=32)

# Evaluate accuracy
test_loss, test_acc = cnn_model.evaluate(test_imgs, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
