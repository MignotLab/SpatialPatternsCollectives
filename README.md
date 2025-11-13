# The mechanisms of spatial pattern transition in motile bacterial collectives

This repository provides the **Python 3** analysis and simulation codes used in the paper:  
**"The mechanisms of spatial pattern transition in motile bacterial collectives" (2025)**  
Saulnier *et al.*, 2025.

## Authors
Jean-Baptiste Saulnier, Michèle Romanos, Jonathan Schrohe, Clémence Cuzin,  
Vincent Calvez*, Tâm Mignot* (*corresponding authors*)

All codes are implemented as *pseudo-notebooks*: each code cell can be executed independently using `Ctrl + Enter`.

---

### 1. Clone this repository

You can clone this public repository using:

```bash
git clone https://github.com/MignotLab/SpatialPatternsCollectives.git
```

---

## 2. Repository structure

All scripts are located in the `scripts/` folder.

### Main analysis scripts

- **`nematic_analysis.py`**  
  Extracts the *nematic order* from phase-contrast or fluorescence movies by dividing each frame into a grid.
  Computes spatial correlations of nematic alignment and plots nematic order as a function of distance *r*.  
  *Used in Figures 1c, 5b, and Supplementary Figure S2c.*

- **`reversal_and_signaling_analysis.py`**  
  Detects reversal events from tracked bacteria (CSV format) and computes signaling-related quantities:
  `reversals`, `cumul_frustration`, `n_neighbours`, and `n_neg_neighbours`.  
  *Used in Figures 1e, 1f, 3b–e.*

- **`tracking_analysis.py`**  
  Performs segmentation, skeletonization, and tracking of rod-shaped bacteria (e.g. *Myxococcus xanthus*).
  Generates tracking CSVs for input to `reversal_and_signaling_analysis.py`.  
  Includes optional Napari visualization.

- **`simulation_1d_cpu.py` / `simulation_1d_cupy.py`**  
  Implements the 1D simulation model presented in the paper.  
  `simulation_1d_cpu.py` runs on any machine (CPU only).  
  `simulation_1d_cupy.py` enables GPU acceleration using CuPy (see below).  
  *Used in Figure 2c.*  
  (Figure 2d was generated from `simulation/pde_model/linearisation/guzzo_model/main.py`.)

### Subdirectories within `scripts/`

- **`analysis_reversals_detection_SgmX/`**  
  Supplementary analyses corresponding to Section 2.2 of the Supplementary Information.

- **`simu_2d/`**  
  Codes for 2D agent-based simulations of swarming and rippling patterns.  
  The main script `agent_based_simulation_script_paper_2024.py` was used for Figures 5–7.

---

## 3. Environment installation

All analyses and simulations can be executed within a single Python environment.  
The easiest setup method uses [`uv`](https://github.com/astral-sh/uv), a modern, fast Python package manager fully compatible with `pip`.

### Installation using `uv` (recommended)
```bash
# Create and activate a local environment
uv venv .venv
source .venv/bin/activate      # Linux/macOS
# or
.venv\Scripts\Activate.ps1     # Windows PowerShell

# Install all required dependencies
uv pip install -r requirements.txt
```
This environment includes all dependencies required to execute every script in the repository.

### Optional: GPU acceleration (CuPy)

Some simulations (e.g., `simulation_1d_cupy.py`) use GPU acceleration through **CuPy**.


**Linux / Windows (NVIDIA GPU)**
``` bash
uv pip install cupy-cuda12x
```

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
