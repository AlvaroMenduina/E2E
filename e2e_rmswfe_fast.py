"""

[FAST MODE] RMS Wavefront Error analysis


Author: Alvaro Menduina
Date: July 2020

Description:
With this script we calculate the RMS Wavefront Error at the detector plane
for the E2E files

This is a FAST MODE version of the e2e_rmswfe.py script.
Instead of looping over all wavelengths and configurations,
and for each case evaluating just 3 field points,
we loop through each configuration and generate a single instance
of the Merit Function and the Raytrace for all wavelengths

With this approach we can calculate the RMS WFE for all 76 configs X 23 wavelengths X 3 Field points
for a given IFU channel in 1 - 1.5 minutes (assuming a pupil sampling N=4, see Zemax Operand RWRE for details)


"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
from time import time


def pupil_sampling_effect(zosapi, sys_mode, ao_modes, spaxel_scale, spaxels_per_slice, grating,
                          files_path, results_path):
    """
    Function to demonstrate the effect of Pupil Sampling
    as defined in the RWRE operand: N is the number of points in a N x N grid
    for each quadrant of the pupil

    Increasing the pupil sampling makes the RMS WFE results more robust
    but it also slows down the calculation

    :param zosapi:
    :param sys_mode:
    :param ao_modes:
    :param spaxel_scale:
    :param spaxels_per_slice:
    :param grating:
    :param files_path:
    :param results_path:
    :return:
    """

    analysis_dir = os.path.join(results_path, 'RMS_WFE')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis = e2e.RMS_WFE_FastAnalysis(zosapi=zosapi)

    options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale],
               'IFUs': ['AB'], 'grating': [grating]}

    N_samp = np.arange(4, 42, 2)
    times = []
    min_rms, mean_rms, max_rms = [], [], []
    rms0 = []

    for pupil_samp in N_samp:

        start = time()
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=None, configuration_idx=[1, 2],
                                                surface=None, spaxels_per_slice=spaxels_per_slice,
                                                pupil_sampling=pupil_samp)
        speed = time() - start
        times.append(speed)
        rms_wfe, obj_xy, foc_xy, waves = list_results[0]
        mean_rms.append(np.mean(rms_wfe))
        min_rms.append(np.min(rms_wfe))
        max_rms.append(np.max(rms_wfe))
        rms0.append(rms_wfe[0, 0, 2])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax.plot(N_samp, times)
    # ax.plot(N_samp, min_rms)
    ax1.plot(N_samp, mean_rms)
    ax1.set_xlabel(r'Pupil Sampling $N$')
    ax1.set_ylabel(r'RMS WFE [nm]')
    ax1.set_title(r'Mean RMS')
    ax1.grid(True)

    m0 = np.mean(mean_rms)
    deltas = (np.array(mean_rms) - m0) / m0 * 100
    ax2.plot(N_samp, deltas)
    ax2.set_xlabel(r'Pupil Sampling $N$')
    ax2.set_ylabel(r'Per cent')
    ax2.set_title(r'Per cent change wrt average')
    ax2.grid(True)

    ax3.plot(N_samp, times)
    ax3.set_xlabel(r'Pupil Sampling $N$')
    ax3.set_ylabel(r'Time [sec]')
    ax3.set_title(r'Computational time')
    ax3.grid(True)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(N_samp, times)
    # ax.plot(N_samp, min_rms)

    plt.show()

    return


def draw_detector_boundary(ax):
    det_pix = 15e-3  # detector pixel in mm
    width = 2 * 4096 * det_pix + 3
    # 2 CCDs + 3 mm gap
    height = 4096 * det_pix
    # Create a Rectangle patch
    rect = Rectangle((-width / 2, -height / 2), width, height, linewidth=1, edgecolor='black',
                     linestyle='--', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    return


def detector_rms_wfe(zosapi, sys_mode, ao_modes, spaxel_scale, spaxels_per_slice, grating, pupil_sampling,
                     files_path, results_path):


    analysis_dir = os.path.join(results_path, 'RMS_WFE')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    analysis = e2e.RMS_WFE_FastAnalysis(zosapi=zosapi)

    rms_maps, object_coord, focal_coord = [], [], []

    for ifu_section in ifu_sections:

        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=None, configuration_idx=None,
                                                surface=None, spaxels_per_slice=spaxels_per_slice,
                                                pupil_sampling=pupil_sampling)
        # Only 1 list, no Monte Carlo
        rms_wfe, obj_xy, foc_xy, waves = list_results[0]

        # print a summary:
        print("\nFor %s scale, IFU-%s, SPEC-%s: " % (spaxel_scale, ifu_section, grating))
        print("RMS: min %.2f | mean %.2f | max %.2f nm " % (np.min(rms_wfe), np.mean(rms_wfe), np.max(rms_wfe)))

        rms_maps.append(rms_wfe)
        focal_coord.append(foc_xy)
        object_coord.append(obj_xy)

    # Stitch the different IFU sections
    rms_field = np.concatenate(rms_maps, axis=1)

    min_rms = np.min(rms_field)
    max_rms = np.max(rms_field)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # Loop over the IFU channels: AB, CD, EF, GH
    for i in range(2):
        for j in range(2):
            k = 2 * i + j
            ifu_section = ifu_sections[k]
            ax = axes[i][j]
            _foc_xy = focal_coord[k]
            _rms_field = rms_maps[k]

            x_odd, y_odd = _foc_xy[::2, :, :, 0].flatten(), _foc_xy[::2, :, :, 1].flatten()
            x_even, y_even = _foc_xy[1::2, :, :, 0].flatten(), _foc_xy[1::2, :, :, 1].flatten()
            triang_odd = tri.Triangulation(x_odd, y_odd)
            triang_even = tri.Triangulation(x_even, y_even)

            min_circle_ratio = .05
            mask_odd = tri.TriAnalyzer(triang_odd).get_flat_tri_mask(min_circle_ratio)
            triang_odd.set_mask(mask_odd)
            mask_even = tri.TriAnalyzer(triang_even).get_flat_tri_mask(min_circle_ratio)
            triang_even.set_mask(mask_even)

            tpc_odd = ax.tripcolor(triang_odd, _rms_field[::2].flatten(), shading='flat', cmap='jet')
            tpc_odd.set_clim(vmin=min_rms, vmax=max_rms)
            tpc_even = ax.tripcolor(triang_even, _rms_field[1::2].flatten(), shading='flat', cmap='jet')
            tpc_even.set_clim(vmin=min_rms, vmax=max_rms)

            draw_detector_boundary(ax)

            axis_label = 'Detector'
            ax.set_xlabel(axis_label + r' X [mm]')
            ax.set_ylabel(axis_label + r' Y [mm]')
            ax.set_aspect('equal')
            cbar = plt.colorbar(tpc_odd, ax=ax, orientation='horizontal')
            cbar.ax.set_xlabel('[nm]')
            title = r'IFU-%s | %s | %s | %s' % (ifu_section, spaxel_scale, grating, sys_mode)
            ax.set_title(title)
            fig_name = "RMSWFE_%s_DETECTOR_SPEC_%s_MODE_%s" % (spaxel_scale, grating, sys_mode)

            save_path = os.path.join(results_path, analysis_dir)
            if os.path.isfile(os.path.join(save_path, fig_name)):
                os.remove(os.path.join(save_path, fig_name))
            fig.savefig(os.path.join(save_path, fig_name))

    N_waves = len(waves)
    rms_maps = np.array(rms_maps)
    object_coord = np.array(object_coord)
    print(object_coord.shape)
    print(rms_maps.shape)
    # Object space coordinates
    for i_ifu, ifu_section in enumerate(ifu_sections):
        for k_wave, wave_str in zip([0, N_waves // 2, -1], ['MIN', 'CENT', 'MAX']):

            fig_obj, ax = plt.subplots(1, 1)

            x_obj, y_obj = object_coord[i_ifu, :, :, 0].flatten(), object_coord[i_ifu, :, :, 1].flatten()
            triang = tri.Triangulation(x_obj, y_obj)
            rms_ = rms_maps[i_ifu, :, k_wave].flatten()
            tpc = ax.tripcolor(triang, rms_, shading='flat', cmap='jet')
            min_rms, max_rms = np.min(rms_), np.max(rms_)
            tpc.set_clim(vmin=min_rms, vmax=max_rms)

            axis_label = 'Object'
            ax.set_xlabel(axis_label + r' X [mm]')
            ax.set_ylabel(axis_label + r' Y [mm]')
            ax.set_aspect('equal')
            cbar = plt.colorbar(tpc, ax=ax, orientation='horizontal')
            cbar.ax.set_xlabel('[nm]')

            wave = waves[k_wave]

            title = r'IFU-%s | %s mas | %s SPEC %.3f $\mu$m | %s %s' % (ifu_section, spaxel_scale, grating, wave, sys_mode, ao_modes[0])
            ax.set_title(title)
            fig_name = "RMSWFE_OBJECT_IFU-%s_%s_SPEC_%s_MODE_%s_%s_WAVE_%s" % (ifu_section, spaxel_scale, grating, sys_mode, ao_modes[0], wave_str)

            save_path = os.path.join(results_path, analysis_dir)
            if os.path.isfile(os.path.join(save_path, fig_name)):
                os.remove(os.path.join(save_path, fig_name))
            fig_obj.savefig(os.path.join(save_path, fig_name))

    return rms_field


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\June_John2020")
    # files_path = os.path.abspath("D:\End to End Model\Monte_Carlo\MedianInstance")
    results_path = os.path.abspath("D:\End to End Model\Results_Report\Mode_NOAO\Scale_4x4")

    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '4x4'
    gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    # gratings = ['H']
    analysis_dir = os.path.join(results_path, 'RMS_WFE')

    # # First we want to justify the choice of Pupil Sampling
    # pupil_sampling_effect(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
    #                            spaxels_per_slice=3, grating='VIS', files_path=files_path, results_path=results_path)

    rms_grating = []
    for grating in gratings:
        rms = detector_rms_wfe(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                               spaxels_per_slice=3, grating=grating, pupil_sampling=4,
                               files_path=files_path, results_path=results_path)
        rms_grating.append(rms.flatten())

    rms_grating = np.array(rms_grating).T

    data = pd.DataFrame(rms_grating, columns=gratings)

    fig_box, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.boxplot(data=data, ax=ax, hue_order=gratings, palette=sns.color_palette("Reds"))
    ax.set_ylabel(r'RMS WFE [nm]')
    ax.set_title(r'RMS WFE Detector | %s scale | %s' % (spaxel_scale, sys_mode))
    # ax.set_ylim([0, 500])

    fig_name = "RMS_WFE_DETECTOR_%s_%s" % (spaxel_scale, sys_mode)
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig_box.savefig(os.path.join(analysis_dir, fig_name))

    plt.show()
