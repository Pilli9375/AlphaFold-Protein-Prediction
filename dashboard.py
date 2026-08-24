import streamlit as st
import pandas as pd
import py3Dmol
from stmol import showmol
import plotly.express as px
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from Bio import PDB
import numpy as np
import io
from pathlib import Path

st.set_page_config(page_title="Protein-Drug Binding Dashboard", layout="wide")

# --------------------------------------------------------------------------------
# AI MODEL DEFINITION & HELPERS
# --------------------------------------------------------------------------------
class GIN(torch.nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64):
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
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GIN(20).to(device)
    model_path = Path('C:/Pilli/DL/Project/trained_protein_model.pth')
    if model_path.exists():
        model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    return model, device

def pdb_to_graph_str(pdb_str):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('uploaded', io.StringIO(pdb_str))
    # Safely get the first available chain
    chain = list(structure[0].get_chains())[0]
    residues = [r for r in chain if r.id[0] == ' ']
    
    x = torch.zeros(len(residues), 20)
    pos = []
    aa_dict = {'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLU':5,'GLN':6,'GLY':7,'HIS':8,'ILE':9,
                 'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,'TRP':17,'TYR':18,'VAL':19}
                 
    for i, r in enumerate(residues):
        if r.resname in aa_dict: 
            x[i, aa_dict[r.resname]] = 1
        if 'CA' in r: 
            pos.append(r['CA'].coord)
            
    pos = np.array(pos)
    edge = []
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            if np.linalg.norm(pos[i]-pos[j]) < 8.0:
                edge.extend([[i,j], [j,i]])
                
    edge = torch.tensor(edge).t().contiguous() if edge else torch.empty((2,0), dtype=torch.long)
    data = Data(x=x, edge_index=edge)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

# Load model globally to avoid reloading overhead
model, device = load_model()

# --------------------------------------------------------------------------------
# CSS & UI STYLING
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0c10, #1f2833);
        color: #c5c6c7;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        margin-bottom: 30px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px 0 rgba(69, 162, 158, 0.2);
        border: 1px solid rgba(69, 162, 158, 0.5);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
        text-shadow: 0 0 15px rgba(102, 252, 241, 0.3);
        background: -webkit-linear-gradient(45deg, #66fcf1, #45a29e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0a0b5;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(102, 252, 241, 0.15) !important;
        box-shadow: 2px 0 15px rgba(0, 0, 0, 0.2);
    }
    
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    
    .stPlotlyChart {
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# SIDEBAR / FILE UPLOAD
# --------------------------------------------------------------------------------
st.sidebar.title("Navigation & Settings")
st.sidebar.markdown("---")
st.sidebar.info("Advanced Graph Neural Network (GIN) interface for molecular docking and protein misfolding analysis.")

st.sidebar.markdown("### Upload Custom Structure")
uploaded_file = st.sidebar.file_uploader("Upload custom .pdb file", type=['pdb'])

is_custom = False
target_name = "1UBQ"
target_desc = "Ubiquitin Structure"
risk_result = "Demo File"
risk_score = 0.0
pdb_string = ""

if uploaded_file:
    try:
        pdb_string = uploaded_file.getvalue().decode("utf-8")
        st.sidebar.success(f"Successfully loaded {uploaded_file.name}!")
        
        # Live Inference
        graph = pdb_to_graph_str(pdb_string).to(device)
        with torch.no_grad():
            raw_pred = model(graph.x, graph.edge_index, graph.batch).item()
        
        # De-normalize (assuming mean=0.55, std=0.35 from test_misfolding.py context)
        mean, std = 0.55, 0.35
        score = raw_pred * std + mean
        confidence = min(99, max(50, int(50 + abs(score-0.55)/0.35*50)))
        
        if score > 1.0:
            risk_result = "HIGH RISK"
        elif score > 0.6:
            risk_result = "MODERATE RISK"
        else:
            risk_result = "LOW RISK"
            
        target_name = uploaded_file.name
        target_desc = f"{risk_result} ({confidence}% conf)"
        risk_score = round(score, 2)
        is_custom = True
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

# --------------------------------------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------------------------------------
st.title("Molecular Interaction Analytics")

tab1, tab2, tab3 = st.tabs(['Live 3D Viewer', 'Docking Analytics', 'Model Architecture'])

with tab1:
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Target Protein</div>
            <div class="metric-value" style="font-size: 2.2rem;">{target_name}</div>
            <p style="margin-top:10px; color:#c5c6c7;">{target_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if is_custom:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Misfolding Score</div>
                <div class="metric-value">{risk_score}</div>
                <p style="margin-top:10px; color:#c5c6c7;">AI Conformation Prediction</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <div class="metric-label">Best Binding Score</div>
                <div class="metric-value">810.29</div>
                <p style="margin-top:10px; color:#c5c6c7;">Tau Protein + Aspirin</p>
            </div>
            """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### Interactive 3D Protein Structure")
        
        view = py3Dmol.view(width=800, height=450)
        
        if is_custom and pdb_string:
            view.addModel(pdb_string, 'pdb')
        else:
            # Fallback to query if nothing uploaded
            view = py3Dmol.view(query='pdb:1UBQ', width=800, height=450)
            
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.setBackgroundColor('#1f2833')
        view.zoomTo()
        showmol(view, height=450, width=800)

with tab2:
    st.markdown("### Top Protein-Drug Interactions")
    
    data = {
        "Interaction": ["Tau Protein + Aspirin", "Tau Protein + Atorvastatin", "Amyloid-Beta (Aβ) + Equanil", "Hemoglobin Alpha + Equanil", "Hemoglobin Beta + Cocaine"],
        "Score": [810.29, 683.23, 640.99, 503.69, 490.28]
    }
    df = pd.DataFrame(data).sort_values(by="Score", ascending=True)
    
    fig = px.bar(df, x="Score", y="Interaction", orientation='h',
                 color="Score", color_continuous_scale="Tealgrn",
                 text="Score")
                 
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c5c6c7', family='Inter'),
        margin=dict(l=20, r=20, t=40, b=40),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Geometric Compatibility Score"),
        yaxis=dict(title="")
    )
    fig.update_traces(textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""
    <div class="glass-card">
        <h3>Architecture Overview</h3>
        <p>The underlying model translates 3D atomic coordinates into graph structures, where nodes represent atoms (or residues) and edges represent bonds (or spatial proximity within 8.0 Å).</p>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <ul>
            <li><strong>Input Dimension:</strong> 20 for Proteins (Amino Acid one-hot encoding) | 9 for Drugs (Atomic properties)</li>
            <li><strong>Layers:</strong> 3 x GINConv layers equipped with Multi-Layer Perceptrons and Batch Normalization.</li>
            <li><strong>Pooling:</strong> Global Add Pooling to aggregate node-level features into a graph-level embedding.</li>
            <li><strong>Training Strategy:</strong> Trained with Mean Squared Error (MSE) over 80 epochs using a 15x data augmentation strategy to ensure robust generalization.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
