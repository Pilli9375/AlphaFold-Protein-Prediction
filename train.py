import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from Bio import PDB
from rdkit import Chem
from rdkit.Chem import AllChem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pdb_to_graph(pdb_file, label=None):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('prot', str(pdb_file))
    chain = structure[0]['A']
    residues = [r for r in chain if r.id[0] == ' ']
    aa_dict = {'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLU':5,'GLN':6,'GLY':7,'HIS':8,'ILE':9,
               'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,'TRP':17,'TYR':18,'VAL':19}
    x = torch.zeros(len(residues), 20)
    ca_pos = []
    for i, res in enumerate(residues):
        if res.resname in aa_dict:
            x[i, aa_dict[res.resname]] = 1
        if 'CA' in res:
            ca_pos.append(res['CA'].coord)
    ca_pos = np.array(ca_pos)
    edge_index = []
    for i in range(len(residues)):
        for j in range(i+1, len(residues)):
            if np.linalg.norm(ca_pos[i] - ca_pos[j]) < 8.0:
                edge_index += [[i,j], [j,i]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.float) if label is not None else None
    return Data(x=x, edge_index=edge_index, y=y)

def mol_to_graph(sdf_file, label=None):
    mol = Chem.MolFromMolFile(str(sdf_file), removeHs=False)
    if mol is None:
        raise ValueError(f"Cannot read SDF: {sdf_file}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    x = torch.zeros(mol.GetNumAtoms(), 9)
    for i, a in enumerate(mol.GetAtoms()):
        x[i,0] = a.GetAtomicNum(); x[i,1] = a.GetDegree(); x[i,2] = a.GetFormalCharge()
        x[i,3] = a.GetChiralTag(); x[i,4] = a.GetTotalNumHs(); x[i,5] = a.GetHybridization().real
        x[i,6] = int(a.GetIsAromatic()); x[i,7] = a.GetMass(); x[i,8] = a.GetImplicitValence()
    edge_index = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    edge_index = edge_index + [[j,i] for i,j in edge_index]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.float) if label is not None else None
    return Data(x=x, edge_index=edge_index, y=y)

PROTEIN_DIR = Path('Datasets/01')
DRUG_DIR    = Path('Datasets/02')

protein_files = sorted(PROTEIN_DIR.glob('*.pdb'))
drug_folders  = sorted([f for f in DRUG_DIR.iterdir() if f.is_dir()])
drug_sdf_paths = [next(f.glob('*.sdf')) for f in drug_folders]

risk_mapping = {'Alpha-Synuclein.pdb': 0.9, 'Amyloid-Beta (Aβ).pdb': 0.8, 'hemoglobin_alpha.pdb': 0.1, 'hemoglobin_beta.pdb': 0.1, 'insulin.pdb': 0.2, 'myoglobin.pdb': 0.1, 'Prion Protein.pdb': 0.8, 'Tau Protein.pdb': 0.9}
protein_labels = [risk_mapping.get(p.name, 0.5) for p in protein_files]
drug_labels    = [1.19, 0.90, -0.07, 2.30, 1.40, 1.20, 0.46, 1.05, 2.10, 1.20, 0.46, 0.80]

protein_dataset = [pdb_to_graph(p, l) for p, l in zip(protein_files, protein_labels)]
drug_dataset    = [mol_to_graph(s, l) for s, l in zip(drug_sdf_paths, drug_labels)]

def normalise(dataset, n_train):
    vals = torch.cat([d.y for d in dataset[:n_train]])
    mean, std = vals.mean(), vals.std() + 1e-6
    for d in dataset:
        d.y = (d.y - mean) / std

split_idx = int(len(protein_dataset) * 0.8)
normalise(protein_dataset, split_idx)
normalise(drug_dataset, 8)

protein_train_loader = DataLoader(protein_dataset[:split_idx], batch_size=1, shuffle=True)
protein_val_loader   = DataLoader(protein_dataset[split_idx:], batch_size=1, shuffle=False)
drug_train_loader    = DataLoader(drug_dataset[:8], batch_size=1, shuffle=True)
drug_val_loader      = DataLoader(drug_dataset[8:10], batch_size=1, shuffle=False)
drug_test_loader     = DataLoader(drug_dataset[10:], batch_size=1, shuffle=False)

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        nn1 = Sequential(Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                         Linear(hidden_dim, hidden_dim), ReLU())
        nn2 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                         Linear(hidden_dim, hidden_dim), ReLU())
        nn3 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                         Linear(hidden_dim, hidden_dim), ReLU())
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.conv3 = GINConv(nn3)
        self.lin   = Linear(hidden_dim, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

protein_model = GIN(20).to(device)
drug_model    = GIN(9).to(device)

p_opt = torch.optim.Adam(protein_model.parameters(), lr=1e-3)
p_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(p_opt, mode='min', factor=0.7, patience=5, min_lr=1e-5)
d_opt = torch.optim.Adam(drug_model.parameters(), lr=1e-3)
d_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(d_opt, mode='min', factor=0.7, patience=5, min_lr=1e-5)

def train_one(model, opt, loader):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y.view(-1,1))
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)

def eval_mae(model, loader):
    model.eval()
    mae = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            mae += (out - data.y.view(-1,1)).abs().mean().item()
    return mae / len(loader.dataset)

print("Training Started".center(80, '='))
print(f"{'Epoch':>6} | {'Protein Train MSE':>16} {'Protein Val MAE':>16} | {'Drug Train MSE':>14} {'Drug Val MAE':>14}")
print("-" * 80)

best_p_mae = float('inf')
best_d_mae = float('inf')

for epoch in range(1, 81):
    p_train_mse = train_one(protein_model, p_opt, protein_train_loader)
    p_val_mae   = eval_mae(protein_model, protein_val_loader)
    p_sch.step(p_val_mae)

    d_train_mse = train_one(drug_model, d_opt, drug_train_loader)
    d_val_mae   = eval_mae(drug_model, drug_val_loader)
    d_sch.step(d_val_mae)

    if p_val_mae < best_p_mae:
        best_p_mae = p_val_mae
        torch.save(protein_model.state_dict(), 'trained_protein_model.pth')
    if d_val_mae < best_d_mae:
        best_d_mae = d_val_mae
        torch.save(drug_model.state_dict(), 'trained_drug_model.pth')

    print(f"{epoch:03d}  | {p_train_mse:16.6f} {p_val_mae:16.6f} | {d_train_mse:14.6f} {d_val_mae:14.6f}")

print("=" * 80)
print(f"Training Complete | Best Protein Val MAE: {best_p_mae:.6f} | Best Drug Val MAE: {best_d_mae:.6f}")
print("Models saved: trained_protein_model.pth & trained_drug_model.pth")
