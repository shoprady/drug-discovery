# -*- coding: utf-8 -*-
"""gat_800k_0903.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14wSD5Be1VZX-NeTSeFdXsqDfMB3wVeFm
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""# Data"""

protein_df = pd.read_csv('/home/aix23606/sojeong/new_drug_ai/data/uniprot_sequence_bert_embedding.tsv', sep='\t')
protein_df

protein_df['sequence_length'] = protein_df['sequence'].apply(len)
max_length_df = protein_df.loc[protein_df.groupby('uniprot_id')['sequence_length'].idxmax()]
max_length_df = max_length_df.drop(columns=['sequence_length'])

protein_df = max_length_df

smiles_df = pd.read_csv('/home/aix23606/sojeong/new_drug_ai/data/BindingDB_IC50.csv')

df = smiles_df[['Smiles', 'Uniprot', 'Standard Relation', 'IC50_nM', 'pIC50']]

df = df.drop_duplicates(subset=['Smiles', 'Uniprot'])
df = df.dropna(subset=['IC50_nM', 'pIC50'])

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

"""# Model: Graph Attention Network

## Functions
"""

def pIC50_to_IC50(pic50_values):
    return 10 ** (9 - pic50_values)

def IC50_to_pIC50(ic50_values):
    return 9 - np.log10(ic50_values)

"""기본 코드"""

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=1, concat=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

def atom_features(atom):
    """Create a feature vector for an atom."""
    return torch.tensor([
        atom.GetAtomicNum(),  
        atom.GetDegree(),  
        atom.GetFormalCharge(),  
        atom.GetNumRadicalElectrons(),  
        atom.GetHybridization(),  
    ], dtype=torch.float)

def smiles_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))

    atom_features_tensor = torch.stack(atom_features_list)

    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)

    return Data(x=atom_features_tensor, edge_index=edge_index, y=torch.tensor([target], dtype=torch.float))

df['Graph'] = df.apply(lambda row: smiles_to_graph(row['Smiles'], row['pIC50']), axis=1)
df = df.dropna(subset=['Graph'])

"""## Implement: criterion = MSE Loss"""

train_loader = DataLoader(df['Graph'].tolist(), batch_size=32, shuffle=True)

data_sample = next(iter(train_loader))
input_dim = data_sample.x.size(1)
hidden_dim = 64
output_dim = 1
model = GATModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 20

train_loader = DataLoader(df['Graph'].tolist(), batch_size=32, shuffle=True)

model.train()
for epoch in range(epochs):
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}')

"""### Test: k-means test data"""

test_df = pd.read_csv('/home/aix23606/sojeong/new_drug_ai/data/selected_test_km.csv')

test_df['Graph'] = test_df.apply(lambda row: smiles_to_graph(row['Smiles'], row['pIC50']), axis=1)
test_df = test_df.dropna(subset=['Graph'])

test_loader = DataLoader(test_df['Graph'].tolist(), batch_size=32, shuffle=False)

model.eval()

with torch.no_grad():
    total_loss = 0
    for data in test_loader:
        output = model(data)
        loss = criterion(output, data.y)
        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')

predictions = []
actuals = []
with torch.no_grad():
    for data in test_loader:
        output = model(data)
        predictions.append(output.cpu().numpy())
        actuals.append(data.y.cpu().numpy())

actuals = np.concatenate([np.array(data.y.cpu().numpy()) for data in test_loader], axis=0)
predictions = np.concatenate([np.array(model(data).cpu().detach().numpy()) for data in test_loader], axis=0)

submit = pd.read_csv('/home/aix23606/sojeong/new_drug_ai/data/sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(np.array(predictions))
submit.head()

# pIC50을 IC50으로 변환하여 성능 평가
actuals_ic50 = pIC50_to_IC50(np.array(actuals))
predictions_ic50 = pIC50_to_IC50(np.array(predictions))

rmse = np.sqrt(mean_squared_error(actuals_ic50, predictions_ic50))
normalized_rmse = rmse / (np.max(actuals_ic50) - np.min(actuals_ic50))

# IC50을 pIC50으로 변환하여 성능 평가
actuals_ic50 = IC50_to_pIC50(actuals_ic50)
predictions_ic50 = IC50_to_pIC50(predictions_ic50)

absolute_error = np.abs(actuals_ic50 - predictions_ic50)
correct_ratio = np.mean(absolute_error <= 0.5)

score = 0.5 * (1 - min(normalized_rmse, 1)) + 0.5 * correct_ratio

print(f"Normalized RMSE (A): {normalized_rmse}")
print(f"Correct Ratio (B): {correct_ratio}")
print(f"Score: {score}")

"""### Submit"""

submit_df = pd.read_csv('/home/aix23606/sojeong/new_drug_ai/data/test.csv')

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Create atom features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))

    atom_features_tensor = torch.stack(atom_features_list)

    # Create edge_index from adjacency matrix
    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)

    # Ensure edge_index is in the correct format (torch_geometric expects edge_index to be (2, num_edges))
    if edge_index.shape[0] == 1:
        edge_index = edge_index.t().contiguous()

    return Data(x=atom_features_tensor, edge_index=edge_index)

submit_df['Graph'] = submit_df.apply(lambda row: smiles_to_graph(row['Smiles']), axis=1)

submit_loader = DataLoader(submit_df['Graph'].tolist(), batch_size=32, shuffle=False)

model.eval()

predictions = []

with torch.no_grad():
    for data in submit_loader:
        output = model(data)
        predictions.append(output.cpu().numpy())

predictions = np.concatenate([np.array(model(data).cpu().detach().numpy()) for data in test_loader], axis=0)

submit = pd.read_csv('/home/aix23606/sojeong/new_drug_ai/data/sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(np.array(predictions))
submit.head()

submit.to_csv('/home/aix23606/sojeong/new_drug_ai/submit_gat_0903.csv', index=False)