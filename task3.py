# task3_handwritten_character_recognition.py

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load MNIST dataset (handwritten digits 0–9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize pixel values (0 to 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Reshape data to add channel dimension (for CNN input)
x_train = x_train.reshape(-1, 28, 28, 1)  # shape: (60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)    # shape: (10000, 28, 28, 1)

# 4. Build CNN model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 output classes (digits 0-9)
])

# 5. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 7. Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")

# 8. Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 9. Save model for future use (optional)
model.save("mnist_cnn_model.h5")
