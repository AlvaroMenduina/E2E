""""
FWHM PSF Analysis

The goal is to calculate the FWHM of the PSF along and across the slices
at the Detector Plane

Author: Alvaro Menduina
Date: June 2020
"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

def fwhm_psf_detector(zosapi, sys_mode, ao_modes, spaxel_scale, grating, N_configs, N_waves, N_rays, files_path, results_path):
    """
    Calculate the FWHM PSF at the Detector Plane
    in both directions, acrosss and along the Slices
    for a given spaxel scale and grating configuration, across all 4 IFU channels
    :param zosapi:
    :param sys_mode:
    :param ao_modes:
    :param spaxel_scale:
    :param grating:
    :param N_waves:
    :param N_rays:
    :param files_path:
    :param results_path:
    :return:
    """

    analysis = e2e.FWHM_PSF_Analysis(zosapi=zosapi)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']

    wavelength_idx = np.linspace(1, 23, N_waves).astype(int)
    # Select the config index. Using all of them takes forever
    # We jump every N_configs
    odd_configs = np.arange(1, 76, 2)[::N_configs]
    even_configs = odd_configs + 1
    configs = list(np.concatenate([odd_configs, even_configs]))
    print("Total Configurations: ", len(configs))

    fx, fy = [], []
    focal_coord = []
    for ifu_section in ifu_sections:
        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}
        focal_plane = e2e.focal_planes[sys_mode][spaxel_scale][ifu_section]['DET']

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=configs,
                                                surface=focal_plane, N_rays=N_rays)

        # Only 1 list, no Monte Carlo
        fwhm, psf_cube, obj_xy, foc_xy, wavelengths = list_results[0]
        fwhm_x, fwhm_y = fwhm[:, :, 0].flatten(), fwhm[:, :, 1].flatten()
        fx.append(fwhm_x)
        fy.append(fwhm_y)

    fx = np.array(fx)
    fy = np.array(fy)

    return fx, fy, focal_coord

def fwhm_all_gratings(zosapi, sys_mode, ao_modes, spaxel_scale, grating_list, N_configs, N_waves, N_rays,
                      files_path, results_path):
    """
    Run the FWHM PSF analysis across all spectral bands


    :param zosapi:
    :param sys_mode:
    :param ao_modes:
    :param spaxel_scale:
    :param grating_list:
    :param N_configs:
    :param N_waves:
    :param N_rays:
    :param files_path:
    :param results_path:
    :return:
    """

    fx_grating, fy_grating = [], []
    minX, minY = [], []
    meanX, meanY = [], []
    maxX, maxY = [], []

    for grating in grating_list:

        fx, fy, focal_coord = fwhm_psf_detector(zosapi=zosapi, sys_mode=sys_mode, ao_modes=ao_modes,
                                                spaxel_scale=spaxel_scale, grating=grating,
                                                N_configs=N_configs, N_waves=N_waves, N_rays=N_rays,
                                                files_path=files_path, results_path=results_path)

        print("\nFor Spectral band %s" % grating)
        print("FWHM PSF results: ")
        min_x, min_y = np.min(fx), np.min(fy)
        mean_x, mean_y = np.mean(fx), np.mean(fy)
        max_x, max_y = np.max(fx), np.max(fy)
        print("Along Slice X  | Min: %.2f microns, Max: %.2f microns" % (min_x, max_x))
        print("Across Slice Y | Min: %.2f microns, Max: %.2f microns" % (min_y, max_y))
        minX.append(min_x)
        meanX.append(mean_x)
        maxX.append(max_x)
        minY.append(min_y)
        meanY.append(mean_y)
        maxY.append(max_y)

        fx_grating.append(fx.flatten())
        fy_grating.append(fy.flatten())

    fx_grating = np.array(fx_grating).T
    fy_grating = np.array(fy_grating).T

    data_x = pd.DataFrame(fx_grating, columns=grating_list)
    data_y = pd.DataFrame(fy_grating, columns=grating_list)

    max_val = np.round(max(np.max(fx_grating), np.max(fy_grating))) + 2
    min_val = np.floor(min(np.min(fx_grating), np.min(fy_grating))) - 2

    fig_box, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    sns.boxplot(data=data_x, ax=ax1, hue_order=grating_list, palette=sns.color_palette("Blues"))
    sns.boxplot(data=data_y, ax=ax2, hue_order=grating_list, palette=sns.color_palette("Reds"))
    ax1.set_ylim([min_val, max_val])
    ax2.set_ylim([min_val, max_val])
    ax1.set_title('FWHM Along Slice [X]')
    ax2.set_title('FWHM Across Slice [Y]')
    ax1.set_ylabel('FWHM [$\mu$m]')
    ax2.set_ylabel('FWHM [$\mu$m]')

    fig_name = "FWHM_PSF_%s_MODE_%s" % (spaxel_scale, sys_mode)
    if os.path.isfile(os.path.join(results_path, fig_name)):
        os.remove(os.path.join(results_path, fig_name))
    fig_box.savefig(os.path.join(results_path, fig_name))

    stats = [minX, meanX, maxX, minY, meanY, maxY]

    return fx_grating, fy_grating, stats


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\June_2020")
    results_path = os.path.abspath("D:\End to End Model\Results_June")

    analysis_dir = os.path.join(results_path, 'FWHM')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    results_path = os.path.join(results_path, analysis_dir)

    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '20x20'
    gratings = ['VIS', 'IZ', 'J', 'IZJ', 'Z_HIGH', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    # gratings = ['H']
    N_rays = 500
    N_waves = 5
    N_configs = 5       # Jump every N_configs, not the total

    fx, fy, stats = fwhm_all_gratings(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes,
                                      spaxel_scale=spaxel_scale, grating_list=gratings,
                                      N_configs=N_configs, N_waves=N_waves, N_rays=N_rays,
                                      files_path=files_path, results_path=results_path)

    plt.show()
