import pandas as pd
import numpy as np
import os
from math import sqrt

from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import load_model

from utils import seed_everything, gen_smiles2graph, pIC50_to_IC50, IC50_to_pIC50, logIC50_to_IC50


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


#===============================================================================
# Load Custom Test Set
#===============================================================================

test = pd.read_csv('./dataset/new_test.csv')
model = load_model("./ckpts/model.h5", custom_objects={'GCNLayer': GCNLayer, 'GRLayer': GRLayer})

def test_example():
    for i in range(len(test)):
        graph = gen_smiles2graph(test.Smiles[i])
        pic = test.pIC50[i]
        if graph[0] is not None or graph[1] is not None:
            yield graph, pic

test_data = tf.data.Dataset.from_generator(
    test_example,
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        (tf.TensorShape([None, 106]), tf.TensorShape([None, None])),
        tf.TensorShape([]),
    ),
)


#===============================================================================
# Model Prediction
#===============================================================================

y_pred = model.predict(test_data.batch(1), verbose=0)[:, 0]
y_true = test.pIC50


#===============================================================================
# Model Evaluation
#===============================================================================

# pIC50
absolute_error = np.abs(y_true - y_pred) # Absolute Error
correct_ratio = np.mean(absolute_error <= 0.5) # Correct Ratio

# IC50
y_pred, y_true = pIC50_to_IC50(y_pred), pIC50_to_IC50(y_true)
rmse = sqrt(mean_squared_error(y_true, y_pred)) # RMSE
normalized_rmse = rmse / (np.max(y_true) - np.min(y_true)) # Normalized RMSE

# Final score
score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * correct_ratio

print(f"Normalized RMSE (A): {normalized_rmse}")
print(f"Correct Ratio (B): {correct_ratio}")
print(f"Score: {score}")


#===============================================================================
# Submission
#===============================================================================

submit_test = pd.read_csv('./open/test.csv')

def submit_example():
    for i in range(len(submit_test)):
        graph = gen_smiles2graph(submit_test.Smiles[i])
        pic = None
        yield graph, pic

submit_data = tf.data.Dataset.from_generator(
    submit_example,
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        (tf.TensorShape([None, 106]), tf.TensorShape([None, None])),
        tf.TensorShape([]),
    ),
)

y_pred = model.predict(submit_data.batch(1), verbose=0)[:, 0]

submit = pd.read_csv('./open/sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(y_pred)
submit.to_csv('./submission/submit_all_data.csv', index=False)