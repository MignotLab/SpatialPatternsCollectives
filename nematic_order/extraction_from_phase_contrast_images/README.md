# Nematic order computation (from phase contrast movie)
We extracted the mean orientation of the bacteria on a grid in the phase contrast image. For each cell of the grid, the "neighbours" were detected within a ring of thickness dr and radius r. The nematic order was averaged for each radius. This operation will be repeated for three other experiments, allowing us to compute the standard deviation.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/MignotLab/nematic_order.git

## Utilisation
Open the extract_nematic_from_phase_contrast.py file, enter the path where the adapted CSV files are saved, and launch it with Ctrl+Enter.