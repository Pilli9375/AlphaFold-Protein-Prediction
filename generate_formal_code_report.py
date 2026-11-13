"""
Formal Source Code Documentation Generator
Generates:
- CSE4006_DL_Project_Code.docx
- CSE4006_DL_Project_Code.pdf

Includes:
1. Title Page with Real VIT-AP Logo (C:\\Pilli\\DL\\Project\\logo.png)
2. Table of Contents
3. Software Architecture & Execution Overview
4. Full, un-truncated annotated source code for all project modules:
   - dashboard.py
   - docking_simulation.py
   - compare_three_models.py
   - generate_metrics.py
   - requirements.txt
"""

import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

PROJECT_DIR = Path(r"C:\Pilli\DL\Project")
os.chdir(str(PROJECT_DIR))

# ==============================================================================
# DOCX XML STYLING HELPERS
# ==============================================================================

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color='D1D5DB', sz='4', val='single'):
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), val)
        border.set(qn('w:sz'), sz)
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), color)
        tblBorders.append(border)
    tblPr.append(tblBorders)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 33, 71) # Deep Oxford Navy
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(15, 60, 115)
    return p

def add_bullet_item(doc, bullet_title, desc):
    """Adds a clean academic bullet item with no text-stretching issues."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    
    r_sym = p.add_run("▪  ")
    r_sym.font.name = 'Arial'
    r_sym.font.size = Pt(8.5)
    r_sym.font.color.rgb = RGBColor(0, 33, 71)

    r_title = p.add_run(bullet_title)
    r_title.font.name = 'Courier New' if '.' in bullet_title else 'Times New Roman'
    r_title.font.size = Pt(10.5)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0, 33, 71)

    r_desc = p.add_run(" " + desc)
    r_desc.font.name = 'Times New Roman'
    r_desc.font.size = Pt(11)
    r_desc.font.color.rgb = RGBColor(40, 40, 40)
    return p

def add_body_paragraph(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(30, 30, 30)
    return p

def add_code_block(doc, file_path):
    """Embeds complete, un-truncated file code inside a styled gray card box."""
    code_path = PROJECT_DIR / file_path
    if not code_path.exists():
        p_err = doc.add_paragraph(f"[ERROR: File {file_path} not found on disk]")
        p_err.runs[0].font.color.rgb = RGBColor(200, 0, 0)
        return

    with open(code_path, 'r', encoding='utf-8', errors='replace') as f:
        code_lines = f.readlines()

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, 'F8F9FA') # Clean, soft off-white / light gray
    set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
    set_table_borders(table, color='D1D5DB', sz='4')

    for i, line in enumerate(code_lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(line.rstrip('\r\n'))
        r.font.name = 'Courier New'
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(33, 37, 41)


# ==============================================================================
# MAIN REPORT GENERATION
# ==============================================================================

def generate_formal_code_report():
    print("==================================================================")
    print("STARTING FORMAL SOURCE CODE REPORT GENERATION")
    print("==================================================================")

    doc = docx.Document()

    # Standard 0.8 inch page margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # --------------------------------------------------------------------------
    # 1. TITLE PAGE
    # --------------------------------------------------------------------------
    p_logo = doc.add_paragraph()
    p_logo.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(12)
    p_logo.paragraph_format.space_after = Pt(16)
    run_logo = p_logo.add_run()
    logo_file = PROJECT_DIR / "logo.png"
    if logo_file.exists():
        run_logo.add_picture(str(logo_file), width=Inches(2.6))
        print("Embedded real logo.png on title page.")

    p_doc_title = doc.add_paragraph()
    p_doc_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_doc_title.paragraph_format.space_before = Pt(10)
    p_doc_title.paragraph_format.space_after = Pt(8)
    r_dt = p_doc_title.add_run("Project Source Code &\nSoftware Architecture Implementation Report")
    r_dt.font.name = 'Times New Roman'
    r_dt.font.size = Pt(19)
    r_dt.font.bold = True
    r_dt.font.color.rgb = RGBColor(0, 33, 71)

    p_proj_title = doc.add_paragraph()
    p_proj_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_proj_title.paragraph_format.space_after = Pt(22)
    r_pt1 = p_proj_title.add_run("Project Title:\n")
    r_pt1.font.name = 'Times New Roman'
    r_pt1.font.size = Pt(12)
    r_pt1.font.bold = True
    r_pt1.font.color.rgb = RGBColor(70, 70, 70)
    r_pt2 = p_proj_title.add_run("Protein Misfolding Risk Prediction & Automated Drug Discovery Using 3D Graph Isomorphism Networks (GIN)")
    r_pt2.font.name = 'Times New Roman'
    r_pt2.font.size = Pt(14)
    r_pt2.font.bold = True
    r_pt2.font.color.rgb = RGBColor(15, 60, 115)

    p_course = doc.add_paragraph()
    p_course.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_course.paragraph_format.space_after = Pt(20)
    r_c1 = p_course.add_run("Course: ")
    r_c1.font.name = 'Times New Roman'
    r_c1.font.size = Pt(12)
    r_c1.font.bold = True
    r_c2 = p_course.add_run("CSE4006 – Deep Learning | Review 3 Evaluation\n")
    r_c2.font.name = 'Times New Roman'
    r_c2.font.size = Pt(13)
    r_c2.font.bold = True
    r_c2.font.color.rgb = RGBColor(0, 33, 71)
    r_c3 = p_course.add_run("Academic Year 2025–2026")
    r_c3.font.name = 'Times New Roman'
    r_c3.font.size = Pt(11.5)

    p_team = doc.add_paragraph()
    p_team.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_team.paragraph_format.space_after = Pt(20)
    r_t_lbl = p_team.add_run("Submitted By:\n")
    r_t_lbl.font.name = 'Times New Roman'
    r_t_lbl.font.size = Pt(12)
    r_t_lbl.font.bold = True

    members = [
        ("V Naga Adithya", "23BCE9062"),
        ("Vanam Harsha Vardhan", "24BCE8190"),
        ("Aadinath Harilal", "23BCE7743")
    ]
    for name, reg in members:
        r_m = p_team.add_run(f"• {name}  ({reg})\n")
        r_m.font.name = 'Times New Roman'
        r_m.font.size = Pt(12)
        r_m.font.bold = True
        r_m.font.color.rgb = RGBColor(20, 20, 20)

    p_guide = doc.add_paragraph()
    p_guide.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide.paragraph_format.space_after = Pt(20)
    r_g1 = p_guide.add_run("Faculty Guide:\n")
    r_g1.font.name = 'Times New Roman'
    r_g1.font.size = Pt(12)
    r_g2 = p_guide.add_run("Prof. Ganji Tejasree\n")
    r_g2.font.name = 'Times New Roman'
    r_g2.font.size = Pt(13.5)
    r_g2.font.bold = True
    r_g2.font.color.rgb = RGBColor(0, 33, 71)
    r_g3 = p_guide.add_run("School of Computer Science and Engineering (SCOPE), VIT-AP University")
    r_g3.font.name = 'Times New Roman'
    r_g3.font.size = Pt(11.5)

    p_repo = doc.add_paragraph()
    p_repo.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_repo.paragraph_format.space_after = Pt(0)
    r_rp1 = p_repo.add_run("Official GitHub Repository:\n")
    r_rp1.font.name = 'Times New Roman'
    r_rp1.font.size = Pt(11)
    r_rp1.font.bold = True
    r_rp2 = p_repo.add_run("https://github.com/Pilli9375/AlphaFold-Protein-Prediction")
    r_rp2.font.name = 'Times New Roman'
    r_rp2.font.size = Pt(11)
    r_rp2.font.color.rgb = RGBColor(0, 102, 204)
    r_rp2.font.underline = True

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # TABLE OF CONTENTS
    # --------------------------------------------------------------------------
    add_heading_1(doc, "Table of Contents")

    toc_table = doc.add_table(rows=1, cols=4)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_widths = [Inches(0.9), Inches(3.9), Inches(1.3), Inches(0.7)]

    hdr_cells = toc_table.rows[0].cells
    for i, h in enumerate(["Section", "Module / Source Code Description", "File Reference", "Page"]):
        hdr_cells[i].width = toc_widths[i]
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i < 2 else WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_cells[i], '002147')
        set_cell_margins(hdr_cells[i], top=70, bottom=70, left=90, right=90)

    toc_items = [
        ("1.0", "Software Architecture, Pipeline Flow & Execution Guidelines", "Overview", "3"),
        ("2.0", "Interactive Streamlit Web Dashboard & Visual Analytics", "dashboard.py", "5"),
        ("3.0", "High-Throughput Geometric Molecular Docking Pipeline", "docking_simulation.py", "17"),
        ("4.0", "Review 3 Multi-Model Comparative Evaluation Suite", "compare_three_models.py", "19"),
        ("5.0", "Analytical Metrics Generation & Statistical Plotting Engine", "generate_metrics.py", "26"),
        ("6.0", "Environment Dependencies & Packaging Manifest", "requirements.txt", "29")
    ]

    for s_num, s_desc, s_file, s_page in toc_items:
        row = toc_table.add_row()
        for idx, text in enumerate([s_num, s_desc, s_file, s_page]):
            cell = row.cells[idx]
            cell.width = toc_widths[idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            if idx == 0 or idx == 3:
                r.font.bold = True
            if idx == 2 and text != "Overview":
                r.font.name = 'Courier New'
                r.font.size = Pt(8.5)
                r.font.color.rgb = RGBColor(0, 102, 153)
            set_cell_background(cell, 'F8FAFC' if len(toc_table.rows) % 2 == 0 else 'FFFFFF')
            set_cell_margins(cell, top=55, bottom=55, left=90, right=90)
            set_cell_background(cell, 'F8FAFC' if len(toc_table.rows) % 2 == 0 else 'FFFFFF')
            set_cell_margins(cell, top=55, bottom=55, left=90, right=90)

    set_table_borders(toc_table, color='D1D5DB', sz='4')
    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 2. SOFTWARE ARCHITECTURE & EXECUTION OVERVIEW
    # --------------------------------------------------------------------------
    add_heading_1(doc, "1. Software Architecture, Pipeline Flow & Execution Guidelines")

    arch_p1 = (
        "The software architecture of the Protein Misfolding Risk Prediction and Automated Drug Discovery Pipeline "
        "is engineered as a modular, end-to-end geometric deep learning platform. Built with PyTorch, PyTorch Geometric, "
        "RDKit, and Streamlit, the system translates physical Cartesian atomic coordinates into non-Euclidean spatial molecular "
        "graphs G = (V, E) to regress conformational risk scores and simulate ligand binding affinities. The codebase is organized "
        "into decoupled functional subsystems ensuring reproducible training, offline deployment, and interactive visualization."
    )
    add_body_paragraph(doc, arch_p1)

    add_heading_2(doc, "1.1 Core Modules & Software Structure")
    
    modules_info = [
        ("dashboard.py:", "Serves as the primary graphical user interface. Implemented via Streamlit, it enables client-side .pdb file uploads, triggers live on-the-fly PyTorch GIN inference, renders interactive 3D structures with py3Dmol WebGL shaders, plots horizontal docking compatibility rankings, and hosts the Review 3 multi-model benchmarking dashboard."),
        ("docking_simulation.py:", "Executes high-throughput geometric docking between target proteins and candidate ligands. Implements RDKit molecular sanitization, applies a 15x Gaussian coordinate perturbation protocol, calculates bounding boxes, and computes docking compatibility scores via SciPy spatial distance matrices."),
        ("compare_three_models.py:", "Implements the empirical Review 3 benchmarking suite. Compares a 3-layer Baseline GCN, a 3-layer Multi-Head GAT, and the Proposed 3D GIN across 80 training epochs, programmatically exporting convergence curves, diagnostic confusion matrices, grouped bar charts, and a structured CSV summary."),
        ("generate_metrics.py:", "Generates high-resolution performance plots for standalone verification, including 80-epoch training MSE loss vs. validation accuracy curves and the dark-themed pathology confusion matrix heatmap."),
        ("requirements.txt:", "Defines the strict Python environment specification, pinning necessary packages including PyTorch, PyTorch Geometric, RDKit, BioPython, Streamlit, py3Dmol, and SciPy.")
    ]

    for mod_title, mod_desc in modules_info:
        add_bullet_item(doc, mod_title, mod_desc)

    # Clean Page Break so Section 1.2 is neatly on Page 4 without awkward splits
    doc.add_page_break()

    add_heading_2(doc, "1.2 Environment Setup & Execution Guidelines")
    
    prereq_text = (
        "The software suite is architected for cross-platform execution across Windows, Linux, and macOS environments "
        "equipped with Python 3.10 or higher. CUDA-compatible GPUs (NVIDIA CUDA 11.8+) are automatically leveraged for accelerated "
        "tensor operations, while an integrated CPU fallback ensures real-time inference in under 200 ms per macromolecular structure."
    )
    add_body_paragraph(doc, prereq_text)

    # Executive Execution Table
    p_tbl_lbl = doc.add_paragraph()
    p_tbl_lbl.paragraph_format.space_before = Pt(8)
    p_tbl_lbl.paragraph_format.space_after = Pt(6)
    p_tbl_lbl.paragraph_format.keep_with_next = True
    r_tbl_lbl = p_tbl_lbl.add_run("Table 1.1: System Setup and Pipeline Execution Reference")
    r_tbl_lbl.font.name = 'Times New Roman'
    r_tbl_lbl.font.size = Pt(11)
    r_tbl_lbl.font.bold = True
    r_tbl_lbl.font.color.rgb = RGBColor(0, 33, 71)

    exec_table = doc.add_table(rows=1, cols=3)
    exec_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    exec_widths = [Inches(1.6), Inches(3.2), Inches(2.0)]

    exec_hdr = exec_table.rows[0].cells
    hdr_titles = ["Phase / Stage", "Terminal CLI Command", "Operational Scope & Output"]
    for i, title in enumerate(hdr_titles):
        exec_hdr[i].width = exec_widths[i]
        p = exec_hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(title)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(exec_hdr[i], '002147')
        set_cell_margins(exec_hdr[i], top=70, bottom=70, left=90, right=90)

    exec_steps = [
        (
            "Step 1: Environment Installation",
            "git clone https://github.com/Pilli9375/AlphaFold-Protein-Prediction.git\ncd AlphaFold-Protein-Prediction\npip install -r requirements.txt",
            "Clones repository and installs PyTorch, PyTorch Geometric, RDKit, BioPython, and dependencies."
        ),
        (
            "Step 2: Launch Interactive Web App",
            "python -m streamlit run dashboard.py --server.port 8502",
            "Launches clinical dashboard on port 8502 with 3D WebGL molecular visualizer and Review 3 tabs."
        ),
        (
            "Step 3: Multi-Model Benchmark",
            "python compare_three_models.py",
            "Evaluates Baseline GCN, Multi-Head GAT, and Proposed 3D GIN over 80 epochs; regenerates charts."
        ),
        (
            "Step 4: Formal Report Compilation",
            "python generate_report.py\npython generate_formal_code_report.py",
            "Compiles academic project report and source code documentation into standardized DOCX and PDF deliverables."
        )
    ]

    for step_title, cmd_text, desc_text in exec_steps:
        row = exec_table.add_row()
        cells = row.cells
        
        # Step cell
        cells[0].width = exec_widths[0]
        p0 = cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r0 = p0.add_run(step_title)
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(9.5)
        r0.font.bold = True
        r0.font.color.rgb = RGBColor(15, 60, 115)
        
        # Command cell
        cells[1].width = exec_widths[1]
        for cmd_idx, line in enumerate(cmd_text.split('\n')):
            p1 = cells[1].paragraphs[0] if cmd_idx == 0 else cells[1].add_paragraph()
            p1.paragraph_format.space_before = Pt(1)
            p1.paragraph_format.space_after = Pt(1)
            p1.paragraph_format.line_spacing = 1.0
            r1 = p1.add_run(line)
            r1.font.name = 'Courier New'
            r1.font.size = Pt(8.5)
            r1.font.color.rgb = RGBColor(30, 41, 59)
        set_cell_background(cells[1], 'F1F5F9')
        
        # Description cell
        cells[2].width = exec_widths[2]
        p2 = cells[2].paragraphs[0]
        p2.paragraph_format.line_spacing = 1.15
        p2.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r2 = p2.add_run(desc_text)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(9.0)
        r2.font.color.rgb = RGBColor(50, 50, 50)
        
        for c_idx, cell in enumerate(cells):
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            if c_idx != 1:
                set_cell_background(cell, 'FFFFFF' if len(exec_table.rows) % 2 == 0 else 'F8FAFC')

    set_table_borders(exec_table, color='D1D5DB', sz='4')

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 3. COMPLETE ANNOTATED SOURCE CODE SECTIONS
    # --------------------------------------------------------------------------

    # Section 2: dashboard.py
    add_heading_1(doc, "2. Streamlit Dashboard & Web Interface (dashboard.py)")
    dash_desc = (
        "The dashboard.py script implements the production-grade interactive web interface for the Protein-Drug pipeline using Streamlit and py3Dmol. "
        "It integrates live PyTorch GIN model inference on user-uploaded .pdb files, renders dynamic 3D molecular structures with WebGL, visualizes "
        "docking interactions with Plotly, and hosts the Review 3 multi-model comparative benchmarking dashboard with direct artifact downloads."
    )
    add_body_paragraph(doc, dash_desc)
    print("Embedding full source code of dashboard.py...")
    add_code_block(doc, "dashboard.py")
    doc.add_page_break()

    # Section 3: docking_simulation.py
    add_heading_1(doc, "3. High-Throughput Geometric Molecular Docking Pipeline (docking_simulation.py)")
    dock_desc = (
        "This module conducts high-throughput geometric docking simulations between target protein conformations and candidate small-molecule ligands. "
        "It leverages RDKit for chemical sanitization and valency validation, extracts spatial bounding boxes, and calculates geometric compatibility scores "
        "based on interatomic contact densities within a 4.5 Å radial cutoff threshold."
    )
    add_body_paragraph(doc, dock_desc)
    print("Embedding full source code of docking_simulation.py...")
    add_code_block(doc, "docking_simulation.py")
    doc.add_page_break()

    # Section 4: compare_three_models.py
    add_heading_1(doc, "4. Review 3 Multi-Model Comparative Evaluation Suite (compare_three_models.py)")
    comp_desc = (
        "This benchmarking suite evaluates and contrasts three distinct graph neural network architectures: Baseline GCN, Multi-Head GAT, and "
        "Proposed 3D GIN on the augmented macromolecular dataset. It programmatically computes classification metrics (Accuracy, Precision, Recall, "
        "F1-Score, MSE, Val MAE) and exports publication-ready convergence curves, side-by-side confusion matrices, and CSV summaries."
    )
    add_body_paragraph(doc, comp_desc)
    print("Embedding full source code of compare_three_models.py...")
    add_code_block(doc, "compare_three_models.py")
    doc.add_page_break()

    # Section 5: generate_metrics.py
    add_heading_1(doc, "5. Analytical Metrics Generation & Statistical Plotting Engine (generate_metrics.py)")
    metric_desc = (
        "This script handles statistical evaluation and performance visualization for the trained Graph Isomorphism Network. "
        "It plots 80-epoch training and validation loss/accuracy convergence curves and computes the pathology confusion matrix "
        "comparing healthy controls against pathogenic misfolded proteins."
    )
    add_body_paragraph(doc, metric_desc)
    print("Embedding full source code of generate_metrics.py...")
    add_code_block(doc, "generate_metrics.py")
    doc.add_page_break()

    # Section 6: requirements.txt
    add_heading_1(doc, "6. Environment Dependencies & Package Manifest (requirements.txt)")
    req_desc = (
        "This file specifies the exact software dependencies and libraries required to reproduce the execution environment. "
        "It pins essential deep learning, chemoinformatics, and visualization packages including PyTorch, PyTorch Geometric, RDKit, "
        "BioPython, and Streamlit."
    )
    add_body_paragraph(doc, req_desc)
    print("Embedding full source code of requirements.txt...")
    add_code_block(doc, "requirements.txt")

    # --------------------------------------------------------------------------
    # SAVE WORD DOCUMENT
    # --------------------------------------------------------------------------
    out_docx = PROJECT_DIR / "CSE4006_DL_Project_Code.docx"
    doc.save(str(out_docx))
    print(f"\n==================================================================")
    print(f"SOURCE CODE REPORT DOCX SAVED SUCCESSFULLY!")
    print(f"Path: {out_docx}")
    print(f"Size: {out_docx.stat().st_size / 1024:.2f} KB")
    print(f"==================================================================")

    # --------------------------------------------------------------------------
    # AUTOMATIC PDF CONVERSION
    # --------------------------------------------------------------------------
    out_pdf = PROJECT_DIR / "CSE4006_DL_Project_Code.pdf"
    pdf_created = False

    print("\nInitiating automatic PDF conversion via docx2pdf...")
    try:
        from docx2pdf import convert
        convert(str(out_docx), str(out_pdf))
        if out_pdf.exists() and out_pdf.stat().st_size > 10000:
            pdf_created = True
            print(f"Successfully created PDF via docx2pdf: {out_pdf}")
    except Exception as e_d2p:
        print(f"docx2pdf issue: {e_d2p}")

    if not pdf_created:
        print("Attempting fallback 1: win32com.client...")
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc_obj = word.Documents.Open(str(out_docx))
            doc_obj.SaveAs(str(out_pdf), FileFormat=17) # 17 = wdFormatPDF
            doc_obj.Close()
            word.Quit()
            if out_pdf.exists() and out_pdf.stat().st_size > 10000:
                pdf_created = True
                print(f"Successfully created PDF via win32com: {out_pdf}")
        except Exception as e_w32:
            print(f"win32com fallback failed: {e_w32}")

    if not pdf_created:
        print("Attempting fallback 2: comtypes.client...")
        try:
            import comtypes.client
            word = comtypes.client.CreateObject("Word.Application")
            word.Visible = False
            doc_obj = word.Documents.Open(str(out_docx))
            doc_obj.SaveAs(str(out_pdf), FileFormat=17)
            doc_obj.Close()
            word.Quit()
            if out_pdf.exists() and out_pdf.stat().st_size > 10000:
                pdf_created = True
                print(f"Successfully created PDF via comtypes: {out_pdf}")
        except Exception as e_ct:
            print(f"comtypes fallback failed: {e_ct}")

    if pdf_created:
        print(f"FINAL CODE PDF READY: {out_pdf} ({out_pdf.stat().st_size / 1024:.2f} KB)")
    else:
        print("WARNING: Could not complete PDF generation automatically.")

if __name__ == "__main__":
    generate_formal_code_report()
