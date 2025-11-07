# %%
# IMPORTANT 🔥
# This guard is essential when using multiprocessing (e.g., with Pool)
# on macOS or Windows. These platforms use the 'spawn' start method, which
# means each subprocess will re-import and re-execute the entire script.
# Without this guard, your main code would run again in every subprocess,
# leading to infinite loops or errors (e.g., RuntimeError: can't pickle...).
# Always wrap your main script logic inside `if __name__ == "__main__":`
if __name__ == "__main__":
    import pandas as pd
    import sys
    import os
    # Ajouter la racine du projet au PYTHONPATH
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from analysis.columns import define_columns
    from analysis.cell_detection import run_detection
    from utils import list_files_in_directory
    from analysis.tracking import Tracker
    import utils as tl


    # Define the number of nodes you want to extract per bacterium
    # Be careful if the number of nodes is too high, then only the longest bacteria will be tracked
    # For example 11 nodes keep object longer than 11 pixels
    n_nodes = 11
    # Call the function to define columns
    COLUMNS_DICT = define_columns(n_nodes)

# You can add multiple folders to process here
    path_folders = [
        "data/segmented_images_100X/",
        # add more folders as needed
        ]
    
    path_folder_saves = [
        'output/tracking/',
        # add same amount of folders as path_folders
        ]
    
    csv_save_filenames = [
        'test_tracking.csv',
        # add same amount of filenames as path_folders
        ]

    # PARAMETERS
    n_jobs = 8

    tbf = 2 / 60 # min / frame
    scale = 0.0646028 # µm / px
    min_size_bacteria = 2  # µm
    max_size_bacteria = 20 # µm

    # Convert min_size_bacteria to pixels
    # and max_velocity to pixels per frame
    max_velocity = 20 # µm / min
    min_size_bacteria_px = min_size_bacteria / scale # px
    max_size_bacteria_px = max_size_bacteria / scale # px
    max_velocity_px_frame = max_velocity * tbf / scale # px / frame

    for path_folder, path_folder_save, csv_save_filename in zip(path_folders, path_folder_saves, csv_save_filenames):
        print(f"Processing folder: {path_folder}")
        # List all image paths in the directory
        ims_all_paths = list_files_in_directory(path_folder)
        df_detection = run_detection(ims_all_paths, n_jobs, min_size_bacteria_px, max_size_bacteria_px, COLUMNS_DICT)
        tr = Tracker(df_detection, COLUMNS_DICT, max_velocity_px_frame)
        df_tracking = tr.track()
        tl.save_dataframe(df_tracking, path_folder_save+csv_save_filename)
        print(f"Tracking results saved to: {path_folder_save+csv_save_filename}")








# %% VISUALIZE TRACKS WITH NAPARI
if __name__ == "__main__":
    import pandas as pd
    import sys
    import os
    from nd2reader import ND2Reader
    import numpy as np
    from skimage.io import imread
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils import list_files_in_directory


    path_folder = "data/segmented_images_100X/"
    path_folder_save = 'output/tracking/'
    csv_save_filename = 'test_tracking.csv'

    path_df = path_folder_save + csv_save_filename
    ims_all_paths = list_files_in_directory(path_folder)  # Liste des chemins des images

    first_frame = 0
    last_frame = 6
    step_ims = 1
    track_df = pd.read_csv(path_df)
    id_col, t_col, x_col, y_col, x_head_col, y_head_col = 'id', 'frame', 'x5', 'y5', 'x10', 'y10'
    point_size = 5


    # stack_masks = None
    stack_masks = [imread(path) for path in ims_all_paths[first_frame:last_frame+1:step_ims]]
    stack_masks = np.stack(stack_masks, axis=0)

    # Image sequence
    # stack_dia = [255 - imread(path_dia + str(i) + ".tif") for i in range(first_frame, last_frame, step_ims)]
    # stack_dia = np.stack(stack_dia, axis=0)

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from analysis.tracking_visualisation import VisualizeTracks
    vt = VisualizeTracks(track_df=track_df,
                         id_col=id_col,
                         t_col=t_col,
                         x_col=x_col,
                         y_col=y_col,
                         point_size=point_size,
                         x_head_col=x_head_col,
                         y_head_col=y_head_col,
                         stack_masks=stack_masks,
                         stack_dia=None, # Allow to visualize on dia images together with masks
                         step_ims=step_ims)

    vt.visualize_tracks()













































# %%
# import matplotlib.pyplot as plt
# t = 10
# df_t = track_df[track_df[t_col] == t]
# seg_to_track_t = df_t.drop_duplicates('id_seg').set_index('id_seg')[id_col].astype(int).to_dict()
# print(seg_to_track_t)

# mask = stack_masks[t]
# track_id_image = np.zeros_like(mask, dtype=np.int32)

# max_seg_id = mask.max()
# lut = np.zeros(max_seg_id + 1, dtype=np.int32)
# for seg_id, track_id in seg_to_track_t.items():
#     if seg_id <= max_seg_id:
#         lut[seg_id] = track_id

# track_id_image = lut[mask]

# print("Tous les seg_id présents dans l'image :", np.unique(mask))
# print("Tous les seg_id associés à un track_id :", list(seg_to_track_t.keys()))


# plt.figure()
# plt.imshow(track_id_image)

# # %%
# import pandas as pd
# import sys
# import os
# from nd2reader import ND2Reader
# import numpy as np
# from skimage.io import imread


# path_folder = '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/segmentation_omnipose/new_seg_paper_2024/test/'
# path_folder_save = '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/segmentation_omnipose/new_seg_paper_2024/test/tracking/'
# csv_save_filename = 'test_tracking.csv'

# path_df = path_folder_save + csv_save_filename
# path_dia = '/Volumes/dock_ssd/DATA/a_postdoc_2024/publication/article_myxo_2024/data_for_submitted_paper/non_formatted_data/fig1/swarming_movie_4.nd2'

# stack_masks = None
# first_frame = 0
# last_frame = 20
# step_ims = 1
# track_df = pd.read_csv(path_df)
# id_col, t_col, x_col, y_col = 'id', 'frame', 'y5', 'x5'
# point_size = 5

# # # nd2
# dia_images = ND2Reader(path_dia)
# stack_dia = [np.asarray(dia_images[i]) for i in range(first_frame, last_frame, step_ims)]
# stack_dia = np.stack(stack_dia, axis=0)

# # Image sequence
# # stack_dia = [255 - imread(path_dia + str(i) + ".tif") for i in range(first_frame, last_frame, step_ims)]
# # stack_dia = np.stack(stack_dia, axis=0)

# # %%
# frame_num = 0  # Choisis ta frame
# df_frame = track_df[track_df['frame'] == frame_num]

# # Compte les occurrences de chaque track_id
# counts = df_frame['id'].value_counts()
# max_id = df_frame['id'].max()
# print('count = ', counts)
# print('max_id = ', max_id)

# # Garde uniquement ceux dont l’occurrence > 1
# duplicates = counts[counts > 1]

# print(f"Track_id avec occurrences > 1 dans la frame {frame_num} :")
# print(duplicates)

# # %%
# import tifffile

# frame0 = tifffile.imread(path_folder+'0.tif')

# import numpy as np
# from skimage.measure import label as connected_label

# bad_labels = []

# # On vérifie chaque label > 0
# for lbl in np.unique(frame0):
#     if lbl == 0:
#         continue  # ignorer le fond

#     # masque binaire pour ce label
#     mask = (frame0 == lbl)

#     # on cherche le nombre de composants connexes dans ce masque
#     conn = connected_label(mask, connectivity=1)
#     n_components = conn.max()

#     if n_components > 1:
#         bad_labels.append((lbl, n_components))

# # Affichage
# if bad_labels:
#     print("⚠️ Certains labels sont fragmentés (plusieurs blobs séparés pour un même ID) :")
#     for lbl, n in bad_labels:
#         print(f" - Label {lbl} a {n} composantes connexes")
# else:
#     print("✅ Tous les labels sont compacts (1 seul blob par ID)")



# %%
