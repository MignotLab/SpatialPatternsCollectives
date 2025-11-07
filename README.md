# The mechanisms of spatial pattern transition in motile bacterial collectives

This repository contains the main analysis and simulation codes used in the paper  
**"The mechanisms of spatial pattern transition in motile bacterial collectives"**.

## Folder: `scripts/`

### **`nematic_analysis.py`**
This script extracts the **nematic order** from phase-contrast or fluorescent movies by dividing each frame into a grid.

- Computes local nematic alignment in each grid bin.
- Correlates nematic order between bins separated by distance *r*.
- The function `plot_nematic()` plots nematic order as a function of distance *r*.

**Used in:** Main Figures **1c**, **5b**, and Supplementary Figure **S2c**.

### **`reversal_and_signaling_analysis.py`**
Detects **cell reversals** from CSV tracking data and computes local signaling features:

- `reversals`: binary flag (1 = reversal at that frame).
- `cumul_frustration`: cumulative local frustration.
- `n_neighbours`: local density.
- `n_neg_neighbours`: directional (nematic) density.

**Used in:** Main Figures **1e**, **1f**, **3b**, **3c**, **3d**, and **3e**.

### **`simulation_1d_main_model.py`**
Runs the **1D simulation** model of the paper.

- Generates spatial pattern transition dynamics in one dimension.
- **Main output:** Figure **2c**.  
  Figure **2d** was generated from  
  `simulation/pde_model/linearisation/guzzo_model/main.py`.

### **`tracking_analysis.py`**
Detects and extracts **cell skeletons** from segmented images, tracks bacterial cells, and generates CSV tracking data for use in `reversal_and_signaling_analysis.py`.

- Visualization supported through **Napari**.
- Works best for **rod-shaped** bacteria such as *Myxococcus xanthus*.

### `analysis_reversals_detection_SgmX/`
Contains analysis codes used for **Section 2.2** of the Supplementary Information.

### `simu_2d/`
Scripts for running **2D agent-based simulations** in parallel for both swarming and rippling behaviors.

- Main script: `agent_based_simulation_script_paper_2024.py`
- **Used in:** Main Figures **5**, **6**, and **7**.

## Usage
All scripts are written as *pseudo-notebooks*: each code cell can be executed independently using `Ctrl + Enter`.

### Create and activate the environment
```bash
uv venv .venv
source .venv/bin/activate      # Linux/macOS
# or
.venv\Scripts\Activate.ps1     # Windows

uv pip install -r requirements.txt
```

### Optional: GPU acceleration (CuPy)

Some 1D simulations use GPU acceleration via **CuPy**.

**Linux/Windows + NVIDIA (CUDA available)**  
Install the CUDA-specific CuPy wheel:
```bash
uv pip install cupy-cuda12x
```