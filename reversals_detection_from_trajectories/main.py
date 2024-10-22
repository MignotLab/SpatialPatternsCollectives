# .py files import
import parameters
import reversals_detection
import cell_direction
import reversal_signal
import tools
import plots

par = parameters.Parameters()
tool = tools.Tools()


# %%
# REVERSALS DETECTION 
# Choose your smoothing parameter
min_size_smoothed_um = 0.5
rev = reversals_detection.ReversalsDetection(min_size_smoothed_um)
rev.reversals_detection()
end_name_file = '_min_size_smoothed_um=' + str(min_size_smoothed_um) + '_um'
tool.save_df(df=rev.smo.df, path=par.path_reversals, file_name='data_rev'+end_name_file+'.csv')
plo = plots.Plots(fontsize=30, end_name_file=end_name_file)
color_rippling = tool.get_rgba_color(color_name='royalblue', alpha=0.4)
color_swarming = tool.get_rgba_color(color_name='limegreen', alpha=0.4)
plo.plot_velocity_distribution(max_vel=15, width_bin=0.5, save=True) # max_vel in µm/min
plo.plot_tbr(min_lifetime=50, color=color_rippling, tbr_max_for_plot=15, ticks_interval_tbr=5, width_bin=20/60, min_tbr=0,save=True)


# DIRECTION EXTRACTION
cdir = cell_direction.CellDirection(end_name_file=end_name_file)
cdir.nodes_directions()
tool.save_df(df=cdir.df, path=par.path_directions, file_name='data_dir'+end_name_file+'.csv')


# SIGNALS DETECTION
sig = reversal_signal.ReversalSignal(end_name_file=end_name_file)
sig.compute_polarity_and_nb_neighbors_angle_view()
sig.compute_polarity_and_nb_neighbors()
tool.save_df(df=sig.df, path=par.path_reversal_signal, file_name='data_rev_sig'+end_name_file+'.csv')