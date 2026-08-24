# 🧬 Protein Misfolding & Drug Discovery Pipeline

Welcome to the **Protein Misfolding & Drug Discovery Pipeline**, a cutting-edge deep learning framework that leverages Graph Neural Networks (GNN) to predict pathogenic protein conformations and screen potential drug candidates via geometric docking simulations.

This project was built to explore the structural behaviors of critical proteins like Tau, Alpha-Synuclein, and Amyloid-Beta, analyzing their misfolding risks and virtually matching them against known pharmaceutical compounds to identify robust binding interactions.

---

## ✨ Key Features

*   **Graph Isomorphism Networks (GIN):** Translates complex 3D atomic coordinates into graph structures (nodes = atoms/residues, edges = spatial proximity) for high-accuracy neural network inference.
*   **15x 3D Data Augmentation:** Prevents model overfitting by injecting random structural perturbations during training, ensuring a highly robust representation of native and misfolded states.
*   **Geometric Docking Simulation:** Utilizes RDKit and SciPy to automatically pair target proteins with drug molecules, calculating a baseline compatibility score based on volumetric ratios, center-of-mass alignment, and spatial proximity.
*   **Live AI Dashboard:** Features an interactive, glassmorphism-styled Streamlit interface. It includes an embedded live 3D molecular viewer and allows users to drag-and-drop `.pdb` files for real-time PyTorch inference.

---

## 🛠 Tech Stack

*   **PyTorch & PyTorch Geometric:** Core deep learning and graph processing engines.
*   **RDKit & BioPython:** Molecular parsing, sanitization, and 3D coordinate extraction for `.sdf` and `.pdb` files.
*   **Streamlit:** Front-end framework powering the sleek, interactive dashboard.
*   **py3Dmol & stmol:** Enabling real-time, interactive 3D rendering of molecules directly in the browser.
*   **Plotly:** Advanced, interactive data visualization for docking analytics.
*   **SciPy:** High-performance spatial algorithms for docking geometry.

---

## 🚀 Installation & Execution

### 1. Install Dependencies
Ensure you have an Anaconda environment active or Python 3.9+ installed. Run the following command to install all necessary requirements:

```bash
pip install torch torch_geometric biopython rdkit streamlit stmol py3Dmol plotly scipy pandas numpy
```

### 2. Launch the Dashboard
To start the live interactive dashboard, navigate to the project directory and execute:

```bash
python -m streamlit run dashboard.py
```

*(Note: Windows users utilizing Anaconda can also double-click the included `start_app.bat` script located in the parent directory to instantly launch the server without manually opening the terminal.)*
