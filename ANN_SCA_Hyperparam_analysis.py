import tensorflow as tf
import numpy as np
import matplotlib
from random import sample
import scipy.io as sio
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Data stored in .mat format
# two fields 
#   traces - array of traces
#   key - array of key used

# devices G2 - G5 are used for training
g2 = sio.loadmat(".\\mat_traces\\cw308XGD2_10k_nov5_1447.mat")
g2_traces = g2['traces'][:, 0:500]
g2_keys = g2['key'][:, 0]

g3 = sio.loadmat(".\\mat_traces\\cw308XGD3_10k_nov5_1643.mat")
g3_traces = g3['traces'][:, 0:500]
g3_keys = g3['key'][:, 0]

g4 = sio.loadmat(".\\mat_traces\\cw308XGD4_10k_nov8_2228.mat")
g4_traces = g4['traces'][:, 0:500]
g4_keys = g4['key'][:, 0]

g5 = sio.loadmat(".\\mat_traces\\cw308XGD5_10k_nov9_1538.mat")
g5_traces = g5['traces'][:, 0:500]
g5_keys = g5['key'][:, 0]

# Devices g6-g9 used for testing cross-device accuracy
g6 = sio.loadmat(".\\mat_traces\\cw308XGD6_10k_nov9_1559.mat")
g6_traces = g6['traces'][:, 0:500]
g6_keys = g6['key'][:, 0]

g8 = sio.loadmat(".\\mat_traces\\cw308XGD8_50k_nov14_1635.mat")
g8_traces = g8['traces'][:, 0:500]
g8_keys = g8['key'][:, 0]

g9 = sio.loadmat(".\\mat_traces\\cw308XGD9_nov14_2011.mat")
g9_traces = g9['traces'][:, 0:500]
g9_keys = g9['key'][:, 0]

# g2 = sio.loadmat(".\\mat_traces\\cw308XGD2_10k_nov5_1447.mat")
# g2_traces = g2['traces'][:, 0:500]
# g2_traces = g2['key'][:, 0]

# Concantenate device traces and keys to get training and testing sets

train_traces = np.vstack([g2_traces, g3_traces, g4_traces, g5_traces])
train_keys = np.hstack([g2_keys, g3_keys, g4_keys, g5_keys])

test_traces = np.vstack([g6_traces, g8_traces, g9_traces])
test_keys = np.hstack([g6_keys, g8_keys, g9_keys])


N = train_traces.shape[0]
train_indices = sample(range(N), int(0.9 * N))
x_train = train_traces[train_indices]
x_test = np.delete(train_traces, train_indices, 0)
y_train = train_keys[train_indices]
y_test = np.delete(train_keys, train_indices, 0)


# samplewise standardization seems mose effective - 6-8% acc on device 2


x_train = (x_train - np.mean(x_train, axis=0)) / (np.std(x_train, axis=0))
x_test = (x_test - np.mean(train_traces[train_indices], axis=0)) / (np.std(train_traces[train_indices], axis=0))
test_traces = (test_traces - np.mean(train_traces[train_indices], axis=0)) / \
    (np.std(train_traces[train_indices], axis=0))

# Hyperparameter sweeps: example of batch size
test_acc = []
for bs in np.arange(16, 1024, 32):
    # Fully Connected Network model
    model_dense = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((500, ), input_shape=(500, )),
        tf.keras.layers.Dense(200, activation=tf.nn.relu,
                              kernel_regularizer=tf.keras.regularizers.l2(0)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(200, activation=tf.nn.relu,
                              kernel_regularizer=tf.keras.regularizers.l2(0)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(256, activation=tf.nn.softmax)
    ])
    model = model_dense
    # Use normal Adam optimizer, default parameters
    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    # Train the model for 10 epochs
    history = model.fit(x_train, y_train, validation_split=0.11,
                        epochs=10, shuffle=True, batch_size=bs)
    # Evaluate same device performance
    metrics = model.evaluate(x_test, y_test)
    # Evaluate cross device performance
    test_metrics = model.evaluate(test_traces, test_keys)
    print(metrics)
    print(test_metrics)
    # save cross device accuracy
    test_acc.append(test_metrics[1])

# Save accuracy results for further analysis
np.save("cross_dev_batch_size_accs.npy", test_acc)

# Plot results here if wanted
plt.rc('font', family='serif', weight='bold')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

x = [r for r in range(16, 1024, 32)]
plt.plot(x, test_acc, linewidth=2)
plt.title('Model Accuracy vs Batch Size on G6')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_xlabel('Batch Size', fontweight='bold')
plt.show()
