import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from pathlib import Path
from rdkit import Chem
import numpy as np

current_dir = Path(__file__).parent.resolve()
model_path = current_dir / "trained_model_composition.pth"
test_folder = current_dir / "test" / "02"

DRUG_NAMES = {
    "1": "Aspirin", "2": "Atorvastatin", "3": "Caffeine", "4": "Cocaine",
    "5": "Diazepam", "6": "Equanil", "7": "Ibuprofen", "8": "Levodopa",
    "9": "Morphine", "10": "Nardil", "11": "Paracetamol", "12": "Penicillin"
}

MEAN_LOGP = 0.9975
STD_LOGP = 1.057

class GIN(torch.nn.Module):
    def __init__(self):
        super(GIN, self).__init__()

        self.conv1 = GINConv(Sequential(
            Linear(9, 64), BatchNorm1d(64), ReLU(),
            Linear(64, 64), ReLU()
        ))
        self.conv2 = GINConv(Sequential(
            Linear(64, 64), BatchNorm1d(64), ReLU(),
            Linear(64, 64), ReLU()
        ))
        self.conv3 = GINConv(Sequential(
            Linear(64, 64), BatchNorm1d(64), ReLU(),
            Linear(64, 64), ReLU()
        ))
        self.lin = Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

def process_pdb(pdb_file):
    mol = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=True)
    if not mol:
        return None
    try:
        Chem.SanitizeMol(mol)
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except:
        return None

    x = []
    for a in mol.GetAtoms():
        x.append([
            a.GetAtomicNum(),
            a.GetDegree(),
            a.GetFormalCharge(),
            a.GetTotalNumHs(),
            int(a.GetHybridization()),
            int(a.GetIsAromatic()),
            a.GetMass() * 0.01,
            a.GetImplicitValence(),
            1.0 if a.GetAtomicNum() in {7, 8, 16} else 0.0
        ])
    x = torch.tensor(x, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if not edges:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)

print("Loading model... ", end="")
device = torch.device('cpu')
model = GIN().to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
model.eval()
print("Done!")

print("\n" + "═" * 70)
print("       DRUG COMPOSITION PREDICTION RESULTS")
print("═" * 70)

with torch.no_grad():
    for folder in sorted(test_folder.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 999):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        drug = DRUG_NAMES.get(folder.name, "Unknown")
        print(f"\n*Sample: {drug}*")

        for pdb in sorted(folder.glob("*.pdb")):
            graph = process_pdb(pdb)
            if graph is None:
                print(f"   {pdb.stem} → Failed")
                continue

            pred = model(graph).item()
            logp = pred * STD_LOGP + MEAN_LOGP
            score = 100 / (1 + np.exp(-(logp - 1.5)))
            score = round(score, 1)

            if score >= 90:
                status = "Outstanding – Highly drug-like"
            elif score >= 80:
                status = "Excellent composition"
            elif score >= 70:
                status = "Very good – suitable"
            elif score >= 50:
                status = "Acceptable"
            else:
                status = "Needs improvement"

            print(f"   {pdb.stem} → Composition Score: {score}% → {status}")

print("\n" + "═" * 70)

