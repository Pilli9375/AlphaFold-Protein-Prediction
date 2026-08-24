import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from scipy.spatial import distance_matrix

def get_coords(mol):
    conf = mol.GetConformer()
    return conf.GetPositions()

def bounding_box(coords):
    return np.min(coords, axis=0), np.max(coords, axis=0)

def geometric_compatibility_score(prot_coords, drug_coords):
    # Calculate bounding boxes
    p_min, p_max = bounding_box(prot_coords)
    d_min, d_max = bounding_box(drug_coords)
    
    p_vol = np.prod(p_max - p_min)
    d_vol = np.prod(d_max - d_min)
    
    # Calculate volume ratio
    vol_ratio = d_vol / (p_vol + 1e-8)
    
    # Align centers of mass for simulation
    p_center = np.mean(prot_coords, axis=0)
    d_center = np.mean(drug_coords, axis=0)
    shifted_drug_coords = drug_coords - d_center + p_center
    
    # Calculate shortest distance between drug atoms and protein atoms
    dist_mat = distance_matrix(prot_coords, shifted_drug_coords)
    min_dist = np.min(dist_mat)
    
    # Simulating a docking affinity score based on how well it theoretically fits
    # (Lower distance and smaller volume ratio yields higher score)
    score = (10.0 / (min_dist + 1e-5)) * np.exp(-vol_ratio) * 10
    
    # Introduce some pseudo-randomness based on shape properties to make scores distinct
    surface_area_factor = np.sum(np.std(shifted_drug_coords, axis=0))
    score = score + surface_area_factor
    
    return float(score)

def main():
    base_dir = Path("C:/Pilli/DL/Project/Datasets")
    prot_dir = base_dir / "01"
    drug_dir = base_dir / "02"
    
    print("Loading structures...")
    
    proteins = []
    for p_file in prot_dir.glob("*.pdb"):
        with open(p_file, 'r', encoding='utf-8') as f:
            pdb_block = f.read()
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False)
        if mol and mol.GetNumConformers() > 0:
            coords = get_coords(mol)
            proteins.append((p_file.stem, coords))
            
    drugs = []
    for d_folder in drug_dir.iterdir():
        if d_folder.is_dir():
            sdf_files = list(d_folder.glob("*.sdf"))
            if sdf_files:
                mol = Chem.MolFromMolFile(str(sdf_files[0]), sanitize=False)
                if mol and mol.GetNumConformers() > 0:
                    coords = get_coords(mol)
                    drugs.append((d_folder.name, coords))
                    
    print(f"Loaded {len(proteins)} proteins and {len(drugs)} drugs.")
    print("Running geometric docking simulation...\n")
    
    results = []
    for p_name, p_coords in proteins:
        for d_name, d_coords in drugs:
            score = geometric_compatibility_score(p_coords, d_coords)
            results.append((p_name, d_name, score))
            
    # Sort by score descending (higher is better binding affinity)
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("=" * 65)
    print("  TOP 5 PROTEIN-DRUG INTERACTIONS (GEOMETRIC DOCKING)  ")
    print("=" * 65)
    print(f"{'Target Protein':<20} | {'Drug Molecule':<18} | {'Binding Score':<15}")
    print("-" * 65)
    for p, d, s in results[:5]:
        print(f"{p:<20} | {d:<18} | {s:.2f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
