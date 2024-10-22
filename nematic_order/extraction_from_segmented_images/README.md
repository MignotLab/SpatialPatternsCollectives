# Nematic order computation
We extracted the orientation of the bacteria from a tracking file that detected the middle, tail, and head of each bacterium. We then computed the nematic order over distance. For each bacterium, the "neighbours" were detected within a ring of thickness dr and radius r. The nematic order was averaged for each radius. This operation will be repeated for three other experiments, allowing us to compute the standard deviation.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/MignotLab/nematic_order.git

## Utilisation
Open the launcher.py file, enter the path where the adapted CSV files are saved, and launch it with Ctrl+Enter.