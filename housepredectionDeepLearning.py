# ---------------------------------------------------
# Deep Learning Project: House Price Prediction
# ---------------------------------------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Random dataset तयार करणे
np.random.seed(42)
X = np.random.rand(100, 1) * 10   # Features: House size (100 houses)
y = 2.5 * X + np.random.randn(100,1) * 2  # Target: House price

# Step 2: Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(1,), activation='relu'),  # Hidden Layer with ReLU
    tf.keras.layers.Dense(1)  # Output Layer
])

# Step 3: Model compile
model.compile(optimizer='adam', loss='mse')

# Step 4: Model summary
model.summary()

# Step 5: Model training
history = model.fit(X, y, epochs=100, batch_size=10, validation_split=0.2)

# Step 6: Predictions & Visualization
y_pred = model.predict(X)

plt.scatter(X, y, label='True Price')
plt.scatter(X, y_pred, label='Predicted Price')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.legend()
plt.show()
