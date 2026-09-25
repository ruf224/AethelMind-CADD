import streamlit as st
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors, Lipinski
from sklearn.ensemble import IsolationForest
import py3Dmol
from stmol import showmol

# Page Configuration Settings
st.set_page_config(page_title="AethelMind-CADD Platform", layout="wide", page_icon="🧬")
st.title("🧬 AethelMind-CADD")
st.subheader("Deep Hybrid Evolutionary AI Molecular Design & Conformational Analysis Suite")
st.write("Generates new molecular configurations optimized via integrated LBDD anomaly filters, simulated SBDD binding pockets, and immediate Lipinski profiling.")

# 1. Ligand-Based Filter Setup (LBDD)
@st.cache_resource
def initialize_lbdd_filter():
    """Trains a baseline structural anomaly detection engine on known drug-like spaces"""
    training_data = [
        [300.0, 2.5, 2, 4, 60.0],
        [400.0, 3.5, 1, 5, 80.0],
        [250.0, 1.5, 3, 3, 50.0],
        [450.0, 4.0, 0, 6, 95.0],
        [350.0, 3.0, 2, 5, 75.0]
    ]
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(training_data)
    return model

lbdd_filter = initialize_lbdd_filter()

# 2. Advanced De Novo Molecular Mutation Engine
def mutate_smiles(base_smi):
    """Mutates structural elements to generate de novo variants within realistic chemical topology"""
    mol = Chem.MolFromSmiles(base_smi)
    if not mol:
        return "CC"
    
    mutation_type = random.choice(["add_carbon", "add_nitrogen", "remove_atom", "add_fluorine", "add_oxygen"])
    try:
        mw = Descriptors.MolWt(mol)
        if mw > 650: # Enforce structural limit checks
            mutation_type = "remove_atom"
            
        editable_mol = Chem.RWMol(mol)
        atoms = list(editable_mol.GetAtoms())
        
        if mutation_type == "add_carbon" and atoms:
            idx = random.choice(range(len(atoms)))
            new_idx = editable_mol.AddAtom(Chem.Atom(6))
            editable_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)
        elif mutation_type == "add_nitrogen" and atoms:
            idx = random.choice(range(len(atoms)))
            new_idx = editable_mol.AddAtom(Chem.Atom(7))
            editable_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)
        elif mutation_type == "add_oxygen" and atoms:
            idx = random.choice(range(len(atoms)))
            new_idx = editable_mol.AddAtom(Chem.Atom(8))
            editable_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)
        elif mutation_type == "add_fluorine" and atoms:
            idx = random.choice(range(len(atoms)))
            new_idx = editable_mol.AddAtom(Chem.Atom(9))
            editable_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)
        elif mutation_type == "remove_atom" and len(atoms) > 4:
            idx = random.choice(range(len(atoms)))
            editable_mol.RemoveAtom(idx)
            
        mutated_mol = editable_mol.GetMol()
        Chem.SanitizeMol(mutated_mol)
        return Chem.MolToSmiles(mutated_mol)
    except:
        return base_smi

# Helper Function: Compute Forcefield Geometries 
def generate_3d_coordinates(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol) # Add hydrogens for proper coordinate optimization
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol) # MMFF94 forcefield refinement simulation
        return Chem.MolToMolBlock(mol)
    except:
        return None

# Check Compliance with Lipinski's Rule of 5
def evaluate_lipinski_compliance(mw, logp, hbd, hba):
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    return "Compliant" if violations <= 1 else f"Non-Compliant ({violations} Violations)"

# 3. Streamlit Sidebar Architecture controls
st.sidebar.header("🔬 Deep Core Synthesis Parameters")
reference_smiles = st.sidebar.text_input("Lead Optimization Target (Base SMILES)", value="c1ccccc1C(=O)O")
num_to_generate = st.sidebar.slider("Generative Batch Iterations Count", min_value=2, max_value=12, value=4)
style_selection = st.sidebar.selectbox("WebGL Conformer Display Style", ["stick", "sphere", "line"])

# Session State initialization for secure persistence across user adjustments
if "leads_data" not in st.session_state:
    st.session_state.leads_data = []

# 4. Pipeline Execution
if st.sidebar.button("Execute Synthesis Pipeline"):
    if not Chem.MolFromSmiles(reference_smiles):
        st.error("Invalid entry for the starter reference SMILES string configuration.")
    else:
        st.info("Synthesizing molecule landscape and screening candidate arrays...")
        
        generated_pool = set()
        current_smi = reference_smiles
        attempts = 0
        
        while len(generated_pool) < (num_to_generate * 5) and attempts < 200:
            current_smi = mutate_smiles(current_smi)
            if Chem.MolFromSmiles(current_smi):
                generated_pool.add(current_smi)
            attempts += 1

        results = []
        for smi in generated_pool:
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue
            
            # Structural Metrics extraction
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            qed_val = QED.qed(mol)
            
            # Lipinski verification calculation
            lipinski_status = evaluate_lipinski_compliance(mw, logp, hbd, hba)
            
            # LBDD Classification
            features = [[mw, logp, hbd, hba, tpsa]]
            score = lbdd_filter.score_samples(features)
            lbdd_pct = min(max(int((score + 1) * 50), 0), 100)
            
            # Simulated SBDD Pocket Affinities
            sbdd_score = -1 * (qed_val * 8.5 + random.uniform(-0.6, 0.6))
            
            results.append({
                "SMILES": smi,
                "QED_DrugLikeness": round(qed_val, 3),
                "SBDD_Affinity_kcal_mol": round(sbdd_score, 2),
                "LBDD_Match": f"{lbdd_pct}%",
                "Lipinski_Status": lipinski_status,
                "Molecular_Weight": round(mw, 2),
                "LogP": round(logp, 2),
                "H_Bond_Donors": hbd,
                "H_Bond_Acceptors": hba
            })
            
        # Select best lead generation hits
        st.session_state.leads_data = sorted(results, key=lambda x: x["QED_DrugLikeness"], reverse=True)[:num_to_generate]

# 5. UI Grid Layout Matrix Component
if st.session_state.leads_data:
    df = pd.DataFrame(st.session_state.leads_data)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    st.subheader("📥 Export Lead Synthesis Results")
    st.download_button(
        label="Download Discovered Drug Leads Dataset (CSV Format)",
        data=csv_bytes,
        file_name="aethelmind_generated_leads.csv",
        mime="text/csv"
    )
    st.divider()

    st.subheader("🚀 Candidate Analytics Grid (3D Spatial Models)")
    
    # Layout grid structure mapping
    cols = st.columns(2)
    for idx, res in enumerate(st.session_state.leads_data):
        with cols[idx % 2]:
            st.markdown(f"### Lead Candidate Candidate #{idx+1}")
            
            # 3D Conformer extraction call
            mol_block_3d = generate_3d_coordinates(res["SMILES"])
            
            if mol_block_3d:
                # WebGL Configuration initialization via stmol/py3Dmol wrappers
                xyz_view = py3Dmol.view(width=450, height=350)
                xyz_view.addModel(mol_block_3d, "mol")
                xyz_view.setStyle({style_selection: {}})
                xyz_view.zoomTo()
                showmol(xyz_view, width=450, height=350)
            else:
                st.warning("⚠️ High structural conformation strain. 3D projection skipped.")
                
            st.text(f"Structure Signature (SMILES): {res['SMILES']}")
            
            # Parameter Data Table layout blocks
            st.markdown(f"""
            *   **Lipinski Evaluation:** `{res['Lipinski_Status']}`
            *   **Quantitative Drug Likeness (QED):** `{res['QED_DrugLikeness']}`
            *   **Molecular Properties:** MW: `{res['Molecular_Weight']}` g/mol | LogP: `{res['LogP']}`
            *   **LBDD Confidence Profile:** `{res['LBDD_Match']}` | **Simulated SBDD Affinity:** `{res['SBDD_Affinity_kcal_mol']} kcal/mol`
            """)
            st.divider()
