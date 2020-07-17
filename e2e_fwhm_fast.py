""""

"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

def draw_detector_boundary(ax):
    """
    Draw a rectangle on the RMS WFE plots
    to indicate the detector boundaries

    Assuming detectors with 4096 pixels, and 3 mm gap in between
    15 micron pixels
    :param ax:
    :return:
    """

    det_pix = 15e-3  # detector pixel in mm
    width = 2 * 4096 * det_pix + 3
    # 2 CCDs + 3 mm gap
    height = 4096 * det_pix
    # Create a Rectangle patch
    rect = Rectangle((-width / 2, -height / 2), width, height, linewidth=1, edgecolor='black',
                     linestyle='--', facecolor='none')
    ax.add_patch(rect)

    return


def fwhm_psf_detector(zosapi, sys_mode, ao_modes, spaxel_scale, grating, N_configs, N_waves, N_rays, files_path, results_path):
    """
    Calculate the FWHM PSF at the Detector Plane (including diffraction effects)
    in both directions, acrosss and along the Slices
    for a given spaxel scale and grating configuration, across all 4 IFU channels

    :param zosapi: the Zemax API
    :param sys_mode: [str] the system mode, either 'HARMONI', 'ELT' or 'IFS
    :param ao_modes: [list] containing the AO mode we want. Only 1 should be used
    :param spaxel_scale: [str] the spaxel scale, ex. '60x30'
    :param grating: [str] the spectral band to analyze, ex. 'HK'
    :param N_waves: how many wavelengths to use. If 3, we will use the min, central and max (typically 5-7)
    :param N_rays: how many pupil rays to trace to estimate the geometric PSF
    :param files_path: path to the E2E files to load for the analysis
    :param results_path: path to save the results
    :return:
    """

    # We will create a separate folder within results_path to save the FWHM results
    analysis_dir = os.path.join(results_path, 'FWHM')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis = e2e.FWHM_PSF_FastAnalysis(zosapi=zosapi)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']

    wavelength_idx = np.linspace(1, 23, N_waves).astype(int)
    # Select the config index. Using all of them takes forever
    # We jump every N_configs
    odd_configs = np.arange(1, 76, 2)[::N_configs]
    even_configs = odd_configs + 1
    # We use both even and odd configurations to cover both A and B paths of the IFU
    configs = list(np.concatenate([odd_configs, even_configs]))
    N = len(configs)
    print("Total Configurations: ", len(configs))

    fx, fy = [], []
    focal_coord, object_coord = [], []
    for ifu_section in ifu_sections:
        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}
        focal_plane = e2e.focal_planes[sys_mode][spaxel_scale][ifu_section]['DET']

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=configs,
                                                surface=focal_plane, N_rays=N_rays)

        # Only 1 list, no Monte Carlo
        fwhm, obj_xy, foc_xy, wavelengths = list_results[0]
        focal_coord.append(foc_xy)
        object_coord.append(obj_xy)
        fwhm_x, fwhm_y = fwhm[:, :, 0], fwhm[:, :, 1]
        fx.append(fwhm_x)
        fy.append(fwhm_y)

    fx = np.array(fx)
    fy = np.array(fy)

    # (1) Detector plane
    # Get a separate figure for each direction X and Y
    for fwhm, label in zip([fx, fy], ['FWHM_X', 'FWHM_Y']):
        min_fwhm = np.nanmin(fwhm)
        max_fwhm = np.nanmax(fwhm)
        # (1) DETECTOR plane plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        # Loop over the IFU channels: AB, CD, EF, GH
        for i in range(2):
            for j in range(2):
                k = 2 * i + j
                ifu_section = ifu_sections[k]
                ax = axes[i][j]
                _foc_xy = focal_coord[k]
                _fwhm = fwhm[k]

                x_odd, y_odd = _foc_xy[:N//2, :, 0].flatten(), _foc_xy[:N//2, :, 1].flatten()
                x_even, y_even = _foc_xy[N//2:, :, 0].flatten(), _foc_xy[N//2:, :, 1].flatten()
                triang_odd = tri.Triangulation(x_odd, y_odd)
                triang_even = tri.Triangulation(x_even, y_even)

                min_circle_ratio = .05
                mask_odd = tri.TriAnalyzer(triang_odd).get_flat_tri_mask(min_circle_ratio)
                triang_odd.set_mask(mask_odd)
                mask_even = tri.TriAnalyzer(triang_even).get_flat_tri_mask(min_circle_ratio)
                triang_even.set_mask(mask_even)

                tpc_odd = ax.tripcolor(triang_odd, _fwhm[:N//2].flatten(), shading='flat', cmap='jet')
                tpc_odd.set_clim(vmin=min_fwhm, vmax=max_fwhm)
                tpc_even = ax.tripcolor(triang_even, _fwhm[N//2:].flatten(), shading='flat', cmap='jet')
                tpc_even.set_clim(vmin=min_fwhm, vmax=max_fwhm)

                draw_detector_boundary(ax)

                axis_label = 'Detector'
                ax.set_xlabel(axis_label + r' X [mm]')
                ax.set_ylabel(axis_label + r' Y [mm]')
                ax.scatter(_foc_xy[:, :, 0].flatten(), _foc_xy[:, :, 1].flatten(), s=2, color='black')
                ax.set_aspect('equal')
                cbar = plt.colorbar(tpc_odd, ax=ax, orientation='horizontal')
                cbar.ax.set_xlabel('[$\mu$m]')
                title = r'%s IFU-%s | %s | %s | %s' % (label, ifu_section, spaxel_scale, grating, ao_modes[0])
                ax.set_title(title)

        fig_name = "%s_%s_DETECTOR_SPEC_%s_MODE_%s_%s" % (label, spaxel_scale, grating, sys_mode, ao_modes[0])

        save_path = os.path.join(results_path, analysis_dir)
        if os.path.isfile(os.path.join(save_path, fig_name)):
            os.remove(os.path.join(save_path, fig_name))
        fig.savefig(os.path.join(save_path, fig_name))

    # # (2) Stitch the Object space coordinates
    # object_coord = np.array(object_coord)
    # x_obj, y_obj = object_coord[:, :, 0].flatten(), object_coord[:, :, 1].flatten()
    # triang = tri.Triangulation(x_obj, y_obj)
    #
    # for k_wave, wave_str in zip([0, N_waves // 2, -1], ['MIN', 'CENT', 'MAX']):
    #
    #     for fwhm, label in zip([fx, fy], ['FWHM_X', 'FWHM_Y']):
    #
    #         min_fwhm = np.nanmin(fwhm)
    #         max_fwhm = np.nanmax(fwhm)
    #
    #         fig_obj, ax = plt.subplots(1, 1)
    #
    #         _fwhm = fwhm[:, :, k_wave].flatten()
    #         tpc = ax.tripcolor(triang, _fwhm, shading='flat', cmap='jet')
    #         tpc.set_clim(vmin=min_fwhm, vmax=max_fwhm)
    #         ax.scatter(x_obj, y_obj, s=2, color='black')
    #
    #         axis_label = 'Object'
    #         ax.set_xlabel(axis_label + r' X [mm]')
    #         ax.set_ylabel(axis_label + r' Y [mm]')
    #         ax.set_aspect('equal')
    #         cbar = plt.colorbar(tpc, ax=ax, orientation='horizontal')
    #         cbar.ax.set_xlabel('[$\mu$m]')
    #
    #         wave = wavelengths[k_wave]
    #
    #         title = r'%s | %s | %s SPEC %.3f $\mu$m | %s %s' % (label, spaxel_scale, grating, wave, sys_mode, ao_modes[0])
    #         ax.set_title(title)
    #         fig_name = "%s_OBJECT_%s_SPEC_%s_MODE_%s_%s_WAVE_%s" % (label, spaxel_scale, grating, sys_mode, ao_modes[0], wave_str)
    #
    #         save_path = os.path.join(results_path, analysis_dir)
    #         if os.path.isfile(os.path.join(save_path, fig_name)):
    #             os.remove(os.path.join(save_path, fig_name))
    #         fig_obj.savefig(os.path.join(save_path, fig_name))

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

    # We will create a separate folder within results_path to save the FWHM results
    analysis_dir = os.path.join(results_path, 'FWHM')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

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
        min_x, min_y = np.nanmin(fx), np.nanmin(fy)
        mean_x, mean_y = np.nanmean(fx), np.nanmean(fy)
        max_x, max_y = np.nanmax(fx), np.nanmax(fy)
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

    # max_val = np.round(max(np.max(fx_grating), np.max(fy_grating))) + 2
    max_valx = np.round(np.nanmax(fx_grating)) + 2
    max_valy = np.round(np.nanmax(fy_grating)) + 5
    min_val = np.floor(min(np.nanmin(fx_grating), np.nanmin(fy_grating))) - 2

    fig_box, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    sns.boxplot(data=data_x, ax=ax1, hue_order=grating_list, palette=sns.color_palette("Blues"))
    sns.boxplot(data=data_y, ax=ax2, hue_order=grating_list, palette=sns.color_palette("Reds"))
    # ax1.set_ylim([min_val, max_valx])
    # ax2.set_ylim([min_val, max_valy])
    ax1.set_title('FWHM Along Slice [X] Detector Plane | %s scale | %s %s' % (spaxel_scale, sys_mode, ao_modes[0]))
    ax2.set_title('FWHM Across Slice [Y] Image Slicer | %s scale | %s %s' % (spaxel_scale, sys_mode, ao_modes[0]))
    ax1.set_ylabel('FWHM [$\mu$m]')
    ax2.set_ylabel('FWHM [$\mu$m]')

    fig_name = "FWHM_PSF_%s_MODE_%s_%s" % (spaxel_scale, sys_mode, ao_modes[0])
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig_box.savefig(os.path.join(analysis_dir, fig_name))

    stats = [minX, meanX, maxX, minY, meanY, maxY]

    return fx_grating, fy_grating, stats


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] This is the bit we have to change when you run the analysis in your system [*]
    sys_mode = 'HARMONI'
    ao_modes = ['LTAO']
    spaxel_scale = '60x30'
    gratings = ['VIS', 'IZ', 'J', 'IZJ', 'Z_HIGH', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    # gratings = ['VIS', 'IZ']
    N_rays = 500
    N_waves = 5
    N_configs = 2  # Jump every N_configs, not the total

    files_path = os.path.abspath("D:\End to End Model\June_John2020")
    results_path = os.path.abspath("D:\End to End Model\Results_Report\Mode_LTAO\Scale_%s" % spaxel_scale)
    # [*] This is the bit we have to change when you run the analysis in your system [*]


    fx, fy, stats = fwhm_all_gratings(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes,
                                      spaxel_scale=spaxel_scale, grating_list=gratings,
                                      N_configs=N_configs, N_waves=N_waves, N_rays=N_rays,
                                      files_path=files_path, results_path=results_path)

    # plt.show()