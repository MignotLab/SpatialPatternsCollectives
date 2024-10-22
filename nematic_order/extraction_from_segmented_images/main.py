# %%
import pandas as pd
import matplotlib.pyplot as plt

import nematic_order
import tools


class Main:


    def __init__(self, inst_par, sample):
        
        self.par = inst_par
        self.tool = tools.Tools()
        self.sample = sample
        self.nem = nematic_order.NematicOrder(self.par, self.sample)


    def start(self):
        """
        Run
        
        """
        print(self.sample+' start')
        # COMPUTE THE NEMATIC ORDER
        df = pd.read_csv(self.par.path_folder+self.par.file_name)

        self.nem.compute_nematic_order(df)

        print(self.sample+' finish')