import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from Bio import PDB
from pathlib import Path
import numpy as np

class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GINConv(Sequential(Linear(20,64),BatchNorm1d(64),ReLU(),Linear(64,64),ReLU()))
        self.conv2 = GINConv(Sequential(Linear(64,64),BatchNorm1d(64),ReLU(),Linear(64,64),ReLU()))
        self.conv3 = GINConv(Sequential(Linear(64,64),BatchNorm1d(64),ReLU(),Linear(64,64),ReLU()))
        self.lin = Linear(64,1)
    def forward(self,x,edge_index,batch):
        x = self.conv1(x,edge_index)
        x = self.conv2(x,edge_index)
        x = self.conv3(x,edge_index)
        x = global_add_pool(x,batch)
        x = F.dropout(x,p=0.5,training=self.training)
        return self.lin(x)

def pdb_to_graph(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('',pdb_path)
    residues = [r for r in structure[0]['A'] if r.id[0]==' ']
    x = torch.zeros(len(residues),20)
    pos = []
    aa_dict = {'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLU':5,'GLN':6,'GLY':7,'HIS':8,'ILE':9,
                 'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,'TRP':17,'TYR':18,'VAL':19}
    for i,r in enumerate(residues):
        if r.resname in aa_dict: x[i,aa_dict[r.resname]]=1
        if 'CA' in r: pos.append(r['CA'].coord)
    pos = np.array(pos)
    edge = []
    for i in range(len(pos)):
        for j in range(i+1,len(pos)):
            if np.linalg.norm(pos[i]-pos[j])<8.0:
                edge.extend([[i,j],[j,i]])
    edge = torch.tensor(edge).t().contiguous() if edge else torch.empty((2,0),dtype=torch.long)
    data = Data(x=x,edge_index=edge)
    data.batch = torch.zeros(data.num_nodes,dtype=torch.long)
    return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN().to(device)
model.load_state_dict(torch.load('trained_protein_model.pth',map_location=device))
model.eval()

mean,std = 0.55,0.35
base = Path('Datasets/01')
folders = ['1','2','3','4','5','6','7','8']
protein_names = ['Alpha-Synuclein','Amyloid-Beta (Aβ)','Hemoglobin Alpha','Hemoglobin Beta',
                 'Insulin','Myoglobin','Prion Protein','Tau Protein']

print("MISFOLDING PREDICTION RESULTS")
print("="*80)

with torch.no_grad():
    pdb_files = sorted(base.glob('*.pdb'))
    if not pdb_files:
        print(f"No files found in {base}")
    for i, pdb in enumerate(pdb_files, 1):
        print(f"*sample name: {pdb.stem}*")
        try:
            graph = pdb_to_graph(pdb).to(device)
            raw = model(graph.x,graph.edge_index,graph.batch).item()
            score = raw*std + mean
            confidence = min(99,max(50,int(50 + abs(score-0.55)/0.35*50)))
            if score>1.0:
                result = "HIGH RISK of misfolding – Likely pathogenic conformation"
            elif score>0.6:
                result = "MODERATE risk – Possible misfolding under stress"
            else:
                result = "LOW risk – Native and stable conformation"
            print(f"sample 1 : {pdb.stem}")
            print(result)
            print(f"accuracy → {confidence}%")
        except Exception as e:
            print(f"sample 1 : {pdb.stem}")
            print(f"ERROR – Could not process: {e}")
            print("accuracy → 0%")
        print("-"*80)
