# FormulaSlug_Projects
Autonomous project 2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


def generate_dataset(n_samples=10000, low=-100.0, high=100.0):
    # 每个样本是 (x1, x2), 标签 y = x1 + x2
    X = np.random.uniform(low, high, size=(n_samples, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)
    return X, y

X, y = generate_dataset(n_samples=20000, low=-100, high=100)
train_frac = 0.8
val_frac = 0.1
n = len(X)
i_train = int(n * train_frac)
i_val = int(n * (train_frac + val_frac))

X_train, y_train = X[:i_train], y[:i_train]
X_val,   y_val   = X[i_train:i_val], y[i_train:i_val]
X_test,  y_test  = X[i_val:], y[i_val:]

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-9
def normalize(x): return (x - mean) / std
X_train_n, X_val_n, X_test_n = normalize(X_train), normalize(X_val), normalize(X_test)

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")  # 回归：linear 输出
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model

model = build_model()
model.summary()

history = model.fit(
    X_train_n, y_train,
    validation_data=(X_val_n, y_val),
    epochs=30,
    batch_size=64,
    verbose=2
)

test_loss, test_mae = model.evaluate(X_test_n, y_test, verbose=0)
print(f"Test MSE: {test_loss:.6f}, Test MAE: {test_mae:.6f}")

def predict_and_print(pairs):
    pairs = np.array(pairs, dtype=np.float32)
    pairs_n = normalize(pairs)
    preds = model.predict(pairs_n).flatten()
    for (a,b), p in zip(pairs, preds):
        print(f"{a:.3f} + {b:.3f} = {p:.3f}  (true {a+b:.3f}, error {p-(a+b):+.3e})")

test_pairs = [[1.2, 3.4], [100, -50], [0.0, 0.0], [-10.5, 2.5], [37.13, 58.87]]
predict_and_print(test_pairs)

model.save("sum_model_tf.keras")
