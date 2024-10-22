# Kymograph computation
We first extracted the position of the bacteria within a specific region defined by an axis (x_i, y_i) to (x_f, y_f), with a certain width (e). We then projected all the positions onto the axis and binarized the number of cells. This process was repeated for several time steps, and the kymograph was plotted.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/MignotLab/kymograph.git

## Utilisation
Open the kymograph.py file, enter the path where the adapted CSV and PNG files are saved, and launch it with Ctrl+Enter.