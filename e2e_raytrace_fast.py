"""



"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
from time import time

def detector_plots(zosapi, sys_mode, ao_modes, spaxel_scale, spaxels_per_slice, grating,
                          files_path, results_path):

    analysis_dir = os.path.join(results_path, 'RAYTRACE')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis = e2e.Raytrace_FastAnalysis(zosapi=zosapi)
    ifu_sections = ['AB', 'CD', 'EF', 'GH']

    N_waves = 5
    wavelength_idx = np.linspace(1, 23, N_waves).astype(int)

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    object_coord, focal_coord = [], []       # Lists to save the results for each IFU channel
    for ifu_section in ifu_sections:

        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=None,
                                                surface=None, spaxels_per_slice=spaxels_per_slice,
                                                ignore_vignetting=True)

        obj_xy, foc_xy, waves = list_results[0]    # Only 1 item on the list, no Monte Carlo files

        focal_coord.append(foc_xy)
        object_coord.append(obj_xy)

        # (1) Detector Focal coordinates plot

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        N_configs = foc_xy.shape[0]
        colors = cm.Reds(np.linspace(0.5, 1.0, N_waves))

        # we need to know which Field point is the inner edge of the slit
        # i.e. obj_x ~ 0.0, this varies between odd and even config
        obj_x = obj_xy[:, :, 0]
        # For each configuration we select the field point closer to f_x = 0
        # by sorting the absolute value of the fx coordinates
        edge_field = [np.argsort(np.abs(obj_x[i]))[0] for i in range(N_configs)]

        for j_wave in range(N_waves):
            x, y = foc_xy[:, j_wave, :, 0].flatten(), foc_xy[:, j_wave, :, 1].flatten()

            # We plot all 3 field points across all configs for a given wavelength
            ax.scatter(x, y, color=colors[j_wave], s=10)

            # add the lambda
            x_max, y_mean = np.max(x), np.min(y)
            ax.text(x=x_max + 2, y=y_mean, s=r'$\lambda_{%d}$' % (wavelength_idx[j_wave]))

            for k_config in range(N_configs):

                # Mark the inner edge of the slit
                i_edge = edge_field[k_config]
                x_edge, y_edge = foc_xy[k_config, j_wave, i_edge, 0], foc_xy[k_config, j_wave, i_edge, 1]
                ax.scatter(x_edge, y_edge, color='black', s=15, facecolors='none')

                # Add a label with the configuration number
                x_c, y_c = foc_xy[k_config, j_wave, 1, 0], foc_xy[k_config, j_wave, 1, 1]
                if k_config >= N_configs//2:
                    ax.text(x=x_c, y=y_c - 1.5, s='%d' % (k_config + 1))
                else:
                    ax.text(x=x_c, y=y_c + 0.5, s='%d' % (k_config + 1))

        ax.set_xlabel('Detector X [mm]')
        ax.set_ylabel('Detector Y [mm]')
        title = r'IFU-%s | %s | %s | %s %s' % (ifu_section, spaxel_scale, grating, sys_mode, ao_modes[0])
        ax.set_title(title)

        fig_name = "CONFIGS_DET_IFU-%s_%s_SPEC_%s_MODE_%s_%s" % (ifu_section, spaxel_scale, grating, sys_mode, ao_modes[0])

        save_path = os.path.join(results_path, analysis_dir)
        if os.path.isfile(os.path.join(save_path, fig_name)):
            os.remove(os.path.join(save_path, fig_name))
        fig.savefig(os.path.join(save_path, fig_name))

        # (2) Object space coordinates

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # we need to know which Field point is the inner edge of the slit
        # i.e. obj_x ~ 0.0, this varies between odd and even config
        obj_x = obj_xy[:, :, 0]
        # For each configuration we select the field point closer to f_x = 0
        # by sorting the absolute value of the fx coordinates
        edge_field = [np.argsort(np.abs(obj_x[i]))[0] for i in range(N_configs)]

        # We plot all 3 field points across all configs for a given wavelength
        x, y = obj_xy[:, :, 0].flatten(), obj_xy[:, :, 1].flatten()
        ax.scatter(x, y, color='red', s=10)

        for k_config in range(N_configs):

            # Mark the inner edge of the slit
            i_edge = edge_field[k_config]
            x_edge, y_edge = obj_xy[k_config, i_edge, 0], obj_xy[k_config, i_edge, 1]
            ax.scatter(x_edge, y_edge, color='black', s=15, facecolors='none')

            # Add a label with the configuration number
            x_c, y_c = obj_xy[k_config, 1, 0], obj_xy[k_config, 1, 1]
            if k_config >= N_configs//2:
                ax.text(x=x_c + 0.02, y=y_c - 0.01, s='%d' % (k_config + 1), fontsize='small')
            else:
                ax.text(x=x_c + 0.02, y=y_c - 0.01, s='%d' % (k_config + 1), fontsize='small')

        ax.set_xlabel('Object X [mm]')
        ax.set_ylabel('Object Y [mm]')
        title = r'IFU-%s | %s | %s | %s %s' % (ifu_section, spaxel_scale, grating, sys_mode, ao_modes[0])
        ax.set_title(title)

        fig_name = "CONFIGS_OBJ_IFU-%s_%s_SPEC_%s_MODE_%s_%s" % (ifu_section, spaxel_scale, grating, sys_mode, ao_modes[0])

        save_path = os.path.join(results_path, analysis_dir)
        if os.path.isfile(os.path.join(save_path, fig_name)):
            os.remove(os.path.join(save_path, fig_name))
        fig.savefig(os.path.join(save_path, fig_name))

    return object_coord, focal_coord


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] This is the bit we have to change when you run the analysis in your system [*]
    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '4x4'
    spaxels_per_slice = 3       # How many field points per Slice to use
    # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    gratings = ['H']
    files_path = os.path.abspath("D:\End to End Model\June_John2020")
    results_path = os.path.abspath("D:\End to End Model\Results_Report\Mode_NOAO\Scale_%s" % spaxel_scale)

    object_xy, focal_xy = [], []
    for grating in gratings:
        obj_xy, foc_xy = detector_plots(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                               spaxels_per_slice=spaxels_per_slice, grating=grating,
                               files_path=files_path, results_path=results_path)
        # object_xy.append(obj_xy)
        # focal_xy.append(foc_xy)

    # for i in range(2):
    #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    #     _foc_xy = foc_xy[i]
    #
    #     N_configs = _foc_xy.shape[0]
    #     N_waves = _foc_xy.shape[1]
    #     configs = np.arange(1, N_configs + 1)
    #     colors = cm.Reds(np.linspace(0.5, 0.75, N_waves))
    #
    #     # we need to know which Field point is the inner edge of the slit
    #     # i.e. obj_x ~ 0.0, this varies between odd and even config
    #     _obj_xy = obj_xy[i]
    #     obj_x = _obj_xy[:, :, 0]
    #     # For each configuration we select the field point closer to f_x = 0
    #     # by sorting the absolute value of the fx coordinates
    #     edge_field = [np.argsort(np.abs(obj_x[i]))[0] for i in range(N_configs)]
    #
    #     for j_wave in range(N_waves):
    #         x, y = _foc_xy[:, j_wave, :, 0].flatten(), _foc_xy[:, j_wave, :, 1].flatten()
    #         ax.scatter(x, y, color=colors[j_wave], s=10)
    #         for k_config in range(N_configs):
    #             # Mark the inner edge of the slit
    #             i_edge = edge_field[k_config]
    #             x_edge, y_edge = _foc_xy[k_config, j_wave, i_edge, 0], _foc_xy[k_config, j_wave, i_edge, 1]
    #             ax.scatter(x_edge, y_edge, color='black', s=15, facecolors='none')
    #
    #             x_c, y_c = _foc_xy[k_config, j_wave, 1, 0], _foc_xy[k_config, j_wave, 1, 1]
    #             ax.text(x=x_c, y=y_c + 0.5, s='%d' % (k_config + 1))


    plt.show()

