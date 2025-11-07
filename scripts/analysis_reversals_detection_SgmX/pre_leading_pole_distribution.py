#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import pickle
import settings

# suggest pole_on_thresh
def get_pole_on_thresh_fct(path_save_image_dir, path_save_dataframe_dir, lead_pole_factor, pole_on_thresh, noise_tol_factor, information_on_off, iteration):

    # import mean fluo intensity values
    if iteration == 0:
        path_analysis = path_save_dataframe_dir + "fluo_analysis.csv"
    else: 
        path_analysis = path_save_dataframe_dir + "fluo_analysis_corrected.csv"

    df = pd.read_csv(path_analysis, index_col = 0)

    # we just do the analysis from the first frame
    cond_t0 = df.loc[:,"t"] == 0
    df = df.loc[cond_t0, :]

    # extract the leading pole and lagging pole intensities
    # note that for now, some of them may be wrongly detected
    cond1 = df.loc[:,"leading_pole"] == 1
    cond2 = df.loc[:,"leading_pole"] == 2
    leadpoles1 = df.loc[cond1, "mean_intensity_pole_1"]
    lagpoles1 = df.loc[cond1, "mean_intensity_pole_2"]
    leadpoles2 = df.loc[cond2, "mean_intensity_pole_2"]
    lagpoles2 = df.loc[cond2, "mean_intensity_pole_1"]
    leading_poles = np.concatenate((leadpoles1,leadpoles2))
    lagging_poles = np.concatenate((lagpoles1,lagpoles2))

    # create simple statistics about their distribution
    mean_leading_poles = np.mean(leading_poles)
    mean_lagging_poles = np.mean(lagging_poles)
    sd_leading_poles = np.std(leading_poles)
    sd_lagging_poles = np.std(lagging_poles)

    # fit to johnson su distibutions:
    try:
        try:
            # skew normal distribution: a bit unaccurate
            #fit_a_lead, fit_loc_lead, fit_scale_lead = stats.skewnorm.fit(leading_poles)
            #fit_a_lag, fit_loc_lag, fit_scale_lag = stats.skewnorm.fit(lagging_poles)

            # assymetric laplace distribution: too spiky
            #fit_kappa_lead, fit_loc_lead, fit_scale_lead = stats.laplace_asymmetric.fit(leading_poles)
            #fit_kappa_lag, fit_loc_lag, fit_scale_lag = stats.laplace_asymmetric.fit(lagging_poles)

            # johnson su distribution:
            fit_a_lead, fit_b_lead, fit_loc_lead, fit_scale_lead = stats.johnsonsu.fit(leading_poles)
            fit_a_lag, fit_b_lag, fit_loc_lag, fit_scale_lag = stats.johnsonsu.fit(lagging_poles)

        except:
            print("Error at fitting process.")

        # calculate a suggestion for the pole_on_thresh.
        # we take the intersection of the two fitted pdfs
        x = np.linspace(np.min(lagging_poles),np.max(leading_poles), 1000)
        
        # skew normal
        #f_lead = stats.skewnorm.pdf(x, a= fit_a_lead, loc = fit_loc_lead, scale= fit_scale_lead)
        #f_lag = stats.skewnorm.pdf(x, a= fit_a_lag, loc = fit_loc_lag, scale= fit_scale_lag)

        # assym laplace
        #f_lead = stats.laplace_asymmetric.pdf(x, kappa= fit_kappa_lead, loc = fit_loc_lead, scale= fit_scale_lead)
        #f_lag = stats.laplace_asymmetric.pdf(x, kappa= fit_kappa_lag, loc = fit_loc_lag, scale= fit_scale_lag)

        # johnson su
        f_lead = stats.johnsonsu.pdf(x, a = fit_a_lead, b = fit_b_lead, loc = fit_loc_lead, scale= fit_scale_lead)
        f_lag = stats.johnsonsu.pdf(x, a = fit_a_lag, b = fit_b_lag, loc = fit_loc_lag, scale= fit_scale_lag)

        intersection_index = np.argwhere(np.diff(np.sign(f_lead - f_lag))).flatten()
        suggestion_pole_on_thresh = x[intersection_index]
        suggestion_pole_on_thresh = suggestion_pole_on_thresh[np.logical_and(suggestion_pole_on_thresh < mean_leading_poles,suggestion_pole_on_thresh > mean_lagging_poles)  ] #since there can be multiple intersections, filter out the correct one

        maximum_index_lag = np.argmax(f_lag) 
        maximum_index_lead = np.argmax(f_lead) 
    except:
        print("Automatic detection of pole_on_thresh failed.")

    # give more information about the algorithm
    if information_on_off == "on":
        print("You can see the distribution of the mean fluorescence values of all leading poles and all lagging poles.\n\
    The decision whether a pole is leading or lagging, has been done with the following three rules:\n\n\
    1. The pole fluo mean has to be above " + str(pole_on_thresh) + " (pole_on_thresh),\n\
    2. the pole has to be bigger than " + str(lead_pole_factor) +  " times bigger than the other (lead_pole_factor)\n\
    3. The pole difference has to be bigger than " + str(noise_tol_factor) + " (noise_tol_factor) times the sd of the mean noise over a summation area. \n\n\
    The pole_on_thresh is changed now to the value at red line!")

        # plot
        fig = plt.figure()
        plt.title("Distribution of leading pole and lagging pole fluo means\n and pole_on_thresh recommendation")
        plt.hist(leading_poles, bins = "auto", label = "leading poles", density = True, alpha = 0.7)
        plt.hist(lagging_poles, bins = "auto", label = "lagging poles", density = True, alpha = 0.6)
        plt.plot(x, f_lead, c = "k")
        plt.plot(x, f_lag, c = "k")
        plt.axvline(suggestion_pole_on_thresh, label = "suggestion pole_on_thresh: " + str( np.round(suggestion_pole_on_thresh,2)), c = "r")
        plt.xlim([ x[maximum_index_lag] - 1.7 * sd_lagging_poles , x[maximum_index_lead] + 4 * sd_leading_poles])

        # new lims just for paper
        plt.ylim([0,2.4])
        plt.xlim([-1,7])

        plt.legend()
        plt.xlabel("Leading poles. Mean: " + str(np.round(mean_leading_poles,2)) + ". SD: " + str(np.round(sd_leading_poles,2)) + \
                "\nLagging poles. Mean: " + str(np.round(mean_lagging_poles,2)) + ". SD: " + str(np.round(sd_lagging_poles,2)))

        
        plt.show()

        fig.savefig(path_save_image_dir + "pole_on_thresh_estimation.svg", format = "svg", bbox_inches='tight')

        data = {"leading_poles": leading_poles, 
                "lagging_poles": lagging_poles, 
                "x": x,
                "fit_leading": f_lead,
                "fit_lagging": f_lag,
                "suggestion_pole_on_thresh": suggestion_pole_on_thresh}
        
        with open(path_save_image_dir + "pole_on_thresh_estimation.pkl", 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

        # %% plot paper
        _size_height_figure = 7
        _figsize = (_size_height_figure, _size_height_figure-1)
        _dpi = 300
        _fontsize = 30
        _fontsize_ticks = _fontsize / 1.5
        _alpha = 0.5
        _color_leading = '#a559aa' # Purple
        _color_lagging = 'grey'
        bin_width = 0.2

        xmin_leading = np.min(leading_poles)
        xmax_leading = np.max(leading_poles)
        step = round((xmax_leading - xmin_leading) / bin_width) + 1
        bins_leading = np.linspace(xmin_leading, xmax_leading, step)

        xmin_lagging = np.min(lagging_poles)
        xmax_lagging = np.max(lagging_poles)
        step = round((xmax_lagging - xmin_lagging) / bin_width) + 1
        bins_lagging = np.linspace(xmin_lagging, xmax_lagging, step)

        min_both = np.minimum(xmin_leading, xmin_lagging)
        max_both = np.maximum(xmax_leading, xmax_lagging)

        fig, ax = plt.subplots(figsize=_figsize)
        ax.hist(leading_poles, 
                bins=bins_leading, 
                label="Leading poles", 
                density=True, 
                alpha=_alpha, 
                color=_color_leading, 
                histtype='bar',
                ec=_color_leading,
                linewidth=0.5)
        ax.hist(lagging_poles, 
                bins=bins_lagging, 
                label="Lagging poles", 
                density=True, 
                alpha=_alpha, 
                color=_color_lagging, 
                histtype='bar',
                ec=_color_lagging,
                linewidth=0.5)
        ax.plot(x, f_lead, c=_color_leading, linewidth=2)
        ax.plot(x, f_lag, c=_color_lagging, linewidth=2)
        ax.axvline(suggestion_pole_on_thresh, label="Threshold: " + str(np.round(suggestion_pole_on_thresh, 2)), c="k")

        ax.set_xlim(min_both, max_both)
        ax.set_xlabel("Mean fluorescence values", fontsize=_fontsize)
        ax.set_ylabel("Density of events", fontsize=_fontsize)
        ax.legend(loc='upper right', handlelength=1, borderpad=0, frameon=False, fontsize=_fontsize_ticks)
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=_fontsize_ticks)
        # ax.set_xticks(np.arange(xmin, xmax+1, step=5))

        fig.savefig(path_save_image_dir + "pole_on_thresh_estimation_paper.png", bbox_inches='tight', dpi=_dpi)
        fig.savefig(path_save_image_dir + "pole_on_thresh_estimation_paper.svg", dpi=_dpi)

    return(suggestion_pole_on_thresh)


