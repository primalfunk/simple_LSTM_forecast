import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
# I haven't figured out why this shows unresolved in VS Code but it is working
from tensorflow.keras.callbacks import TensorBoard, Callback


# Generate sales data: 1000 days
np.random.seed(0)
n = 1000
t = np.linspace(0, 4 * np.pi, n)
y = 20 * np.sin(t) + 50 + 5 * np.random.normal(size=len(t))  # Seasonal + Trend + Noise

df = pd.DataFrame({'Sales': y})
df.to_csv('synthetic_sales_data.csv', index=False)

# Load data
df = pd.read_csv('synthetic_sales_data.csv')
data = df['Sales'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare sequences
seq_len = 10
X, y = [], []
for i in range(len(data) - seq_len):
    X.append(data_scaled[i:i+seq_len])
    y.append(data_scaled[i+seq_len])

X = np.array(X)
y = np.array(y)

# Train/Test Split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

class CustomCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        print(f"Finished batch {batch}, Batch Loss: {logs['loss']}")


tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, 
          y_train, 
          epochs=50, 
          batch_size=64,
          verbose=1,
          callbacks=[CustomCallback(), tensorboard])

# Evaluate the model
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
y_true = scaler.inverse_transform(y_test).flatten()

mae = mean_absolute_error(y_true, y_pred)
print(f'Mean Absolute Error: {mae}')