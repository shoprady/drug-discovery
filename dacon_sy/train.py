import pandas as pd
import os

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import seed_everything, gen_smiles2graph, pIC50_to_IC50


#===============================================================================
# Set up GPU and Seed
#===============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

CFG = {
    'NBITS': 2048,
    'SEED': 42,
}

seed_everything(CFG['SEED'])  # Seed 고정


#===============================================================================
# Load and Split Data
#===============================================================================

chembl_data = pd.read_csv('./dataset/new_train.csv')

def example():
    for i in range(len(chembl_data)):
        graph = gen_smiles2graph(chembl_data.Smiles[i])
        pic = chembl_data.pIC50[i]
        yield graph, pic

data = tf.data.Dataset.from_generator(
    example,
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        (tf.TensorShape([None, 106]), tf.TensorShape([None, None])),
        tf.TensorShape([]),
    ),
)

val_data = data.take(74700)
train_data = data.skip(74700)


#===============================================================================
# Model Implementation
#===============================================================================

class GCNLayer(tf.keras.layers.Layer):
    """Implementation of GCN as layer"""

    def __init__(self, activation=None, **kwargs):
        # constructor, which just calls super constructor
        # and turns requested activation into a callable function
        super(GCNLayer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # create trainable weights
        node_shape, adj_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], node_shape[2]), name="w")

    def call(self, inputs):
        # split input into nodes, adj
        nodes, adj = inputs
        # compute degree
        degree = tf.reduce_sum(adj, axis=-1)
        # GCN equation
        new_nodes = tf.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.w)
        out = self.activation(new_nodes)

        return out, adj

class GRLayer(tf.keras.layers.Layer):
    """A GNN layer that computes average over all node features"""

    def __init__(self, name="GRLayer", **kwargs):
        super(GRLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        nodes, adj = inputs
        # compute the mean over the nodes
        reduction = tf.reduce_mean(nodes, axis=1)
        return reduction

ninput = tf.keras.Input(
    (
        None,
        106,
    )
)

ainput = tf.keras.Input(
    (
        None,
        None,
    )
)

# GCN block with Dropout
x = GCNLayer("relu")([ninput, ainput])
x = GCNLayer("relu")(x)
x = GCNLayer("relu")(x)
x = GCNLayer("relu")(x)

# reduce to graph features
x = GRLayer()(x)

# standard layers (the readout) with Dropout
x = tf.keras.layers.Dense(16, "tanh")(x)
x = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=(ninput, ainput), outputs=x)


#===============================================================================
# Custom Loss
#===============================================================================

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def call(self, y_true, y_pred):
        # pIC50 절대 오차 계산
        absolute_error = tf.abs(y_true - y_pred)
        correct_ratio = tf.reduce_mean(tf.cast(absolute_error <= 0.5, tf.float32))

        # IC50로 변환
        y_pred_ic50 = pIC50_to_IC50(y_pred)
        y_true_ic50 = pIC50_to_IC50(y_true)

        # RMSE 계산
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true_ic50 - y_pred_ic50)))
        normalized_rmse = rmse / (tf.reduce_max(y_true_ic50) - tf.reduce_min(y_true_ic50))

        # Final score
        score = 0.5 * (1 - tf.minimum(normalized_rmse, 1.0)) + 0.5 * correct_ratio

        # loss로 변환
        loss = 1.0 - score

        return score


#===============================================================================
# Training
#===============================================================================

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="mean_squared_error")
es = EarlyStopping(patience=10)
mc = ModelCheckpoint("./ckpts/model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
result = model.fit(train_data.batch(1), validation_data=val_data.batch(1), epochs=100, batch_size=32, callbacks=[es, mc])


#===============================================================================
# Loss Plot
#===============================================================================

plt.plot(result.history['loss'], label="training")
plt.plot(result.history['val_loss'], label="validation")
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./train_val_loss.png')