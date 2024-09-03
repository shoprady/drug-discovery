import numpy as np
import os
import random

from rdkit import Chem
import rdkit

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def gen_smiles2graph(sml):
    """Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    """
    try:
        m = rdkit.Chem.MolFromSmiles(sml)
        m = rdkit.Chem.AddHs(m)
        order_string = {
            rdkit.Chem.rdchem.BondType.SINGLE: 1,
            rdkit.Chem.rdchem.BondType.DOUBLE: 2,
            rdkit.Chem.rdchem.BondType.TRIPLE: 3,
            rdkit.Chem.rdchem.BondType.AROMATIC: 4,
            rdkit.Chem.rdchem.BondType.DATIVE: 5,
        }
        N = len(list(m.GetAtoms()))
        nodes = np.zeros((N, 106))
        for i in m.GetAtoms():
            nodes[i.GetIdx(), i.GetAtomicNum()] = 1

        adj = np.zeros((N, N))
        for j in m.GetBonds():
            u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
            v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
            order = j.GetBondType()
            if order in order_string:
                order = order_string[order]
            else:
                raise Warning("Ignoring bond order", order)
            adj[u, v] = 1
            adj[v, u] = 1
        adj += np.eye(N)
        return nodes, adj

    except Exception as e:
        print(f"Error processing SMILES '{sml}': {e}")
        return None, None

def pIC50_to_IC50(pic50_values):
    return 10 ** (9 - pic50_values)

def IC50_to_pIC50(ic50_values):
    return 9 - np.log10(ic50_values)

def logIC50_to_IC50(logIC50_values):
    return np.expm1(logIC50_values)