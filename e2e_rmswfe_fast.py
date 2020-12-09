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

import json
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

    With this function we plot variations in the RMS WFE results as a function of
    the pupil sampling N, to inform the decision on which N to use, but also
    to get an idea of the variability / error attached

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

    # E2E model files options dictionary. Needed to create a list of the E2E files we have to load for analysis
    options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale],
               'IFUs': ['AB'], 'grating': [grating]}

    # We will try multiple values of N, the pupil sampling and record
    # (1) The time it takes to compute the RMS WFE
    # (2) The minimum, mean and maximum value of the RMS WFE
    # (3) The RMS WFE at a fixed config and wave, to avoid blending away variations with the mean calculation
    N_samp = np.arange(4, 26)
    times = []
    min_rms, mean_rms, max_rms = [], [], []
    rms0 = []

    for pupil_samp in N_samp:

        # For speed, we will only run the analysis for a single IFU channel, and 2 configurations
        start = time()
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=None, configuration_idx=[1, 2],
                                                surface=None, spaxels_per_slice=spaxels_per_slice,
                                                pupil_sampling=pupil_samp, save_hdf5=False)
        # Note that here we specify that it should NOT save the HDF5 files

        speed = time() - start
        times.append(speed)
        rms_wfe, obj_xy, foc_xy, waves = list_results[0]
        mean_rms.append(np.mean(rms_wfe))
        min_rms.append(np.min(rms_wfe))
        max_rms.append(np.max(rms_wfe))
        # Select the first config, first wavelength, and central field point
        rms0.append(rms_wfe[0, 0, 1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    ax1.plot(N_samp, mean_rms, color='black')
    ax1.scatter(N_samp, mean_rms, s=5, color='black')
    ax1.set_xlabel(r'Pupil Sampling $N$')
    ax1.set_ylabel(r'RMS WFE [nm]')
    ax1.set_title(r'Mean RMS')
    ax1.grid(True)

    m0 = np.mean(mean_rms)
    deltas = (np.array(mean_rms) - m0) / m0 * 100
    ax2.plot(N_samp, deltas, color='black')
    ax2.scatter(N_samp, deltas, s=5, color='black')
    ax2.set_xlabel(r'Pupil Sampling $N$')
    ax2.set_ylabel(r'Per cent')
    ax2.set_title(r'Per cent change wrt average')
    ax2.grid(True)

    ax3.plot(N_samp, times, color='black')
    ax3.scatter(N_samp, times, s=5, color='black')
    ax3.set_xlabel(r'Pupil Sampling $N$')
    ax3.set_ylabel(r'Time [sec]')
    ax3.set_title(r'Computational time')
    ax3.grid(True)

    fig_name = "PUP_SAMP_RMSWFE_%s_SPEC_%s_MODE_%s_%s_MeanValue" % (spaxel_scale, grating, sys_mode, ao_modes[0])

    save_path = os.path.join(results_path, analysis_dir)
    if os.path.isfile(os.path.join(save_path, fig_name)):
        os.remove(os.path.join(save_path, fig_name))
    fig.savefig(os.path.join(save_path, fig_name))

    return N_samp, mean_rms, times


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


def detector_rms_wfe(zosapi, file_options, spaxels_per_slice, pupil_sampling, files_path, results_path):
    """
    Calculate the RMS WFE for a given system mode (typically HARMONI), AO mode, spaxel scale
    and spectral band, at the DETECTOR PLANE

    We loop over the IFU channel and calculate, for each E2E file, the RMS WFE for all wavelengths and configurations
    We can specify the number of points per slice [spaxels per slice] to use for the calculation (typically 3)
    The results are plotted:
        (1) at the DETECTOR plane for all wavelengths
        (2) at the OBJECT plane (stitching all IFU channels) for the Min, Central and Max wavelength

    :param zosapi: the Zemax API
    :param sys_mode: [str] the system mode, either 'HARMONI', 'ELT' or 'IFS
    :param ao_modes: [list] containing the AO mode we want. Only 1 should be used
    :param spaxel_scale: [str] the spaxel scale, ex. '60x30'
    :param spaxels_per_slice: number of field points per slice to use for the calculation
    :param grating: [str] the spectral band to analyze, ex. 'HK'
    :param pupil_sampling: the pupil sampling for the RWRE operand, representing a grid of N x N per pupil quadrant
    :param files_path: path to the E2E files to load for the analysis
    :param results_path: path to save the results
    :return:
    """

    spaxel_scale = file_options['SPAX_SCALE']
    grating = file_options['GRATING']
    monte_carlo = file_options['MONTE_CARLO']
    ao_mode = file_options['AO_MODE']

    # We will create a separate folder within results_path to save the RMS WFE results
    analysis_dir = os.path.join(results_path, 'RMS_WFE')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    # ifu_sections = ['AB', 'CD', 'EF', 'GH']
    ifu_sections = ['AB', 'CD']
    analysis = e2e.RMS_WFE_FastAnalysis(zosapi=zosapi)      # The analysis object

    rms_maps, object_coord, focal_coord = [], [], []        # Lists to save the results for each IFU channel
    for ifu_section in ifu_sections:

        # change the IFU path option
        file_options['IFU_PATH'] = ifu_section

        if monte_carlo is True:
            # read the MC instance number of the specific IFU path
            ifu_mc = file_options['IFU_PATHS_MC'][ifu_section]
            file_options['IFU_MC'] = ifu_mc

        # options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
        #            'grating': [grating]}

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=file_options, results_path=results_path,
                                                wavelength_idx=None, configuration_idx=None,
                                                surface=None, spaxels_per_slice=spaxels_per_slice,
                                                pupil_sampling=pupil_sampling, remove_slicer_aperture=True,
                                                monte_carlo=monte_carlo)

        rms_wfe, obj_xy, foc_xy, waves = list_results[0]    # Only 1 item on the list, no Monte Carlo files
        # print(rms_wfe.shape)

        # print a summary to spot any issues:
        print("\nFor %s scale, IFU-%s, SPEC-%s: " % (spaxel_scale, ifu_section, grating))
        print("RMS: min %.2f | mean %.2f | max %.2f nm " % (np.min(rms_wfe), np.mean(rms_wfe), np.max(rms_wfe)))

        # import matplotlib.cm as cm
        # colors = cm.Reds(np.linspace(0.5, 1.0, 76))
        # for j in range(23):
        #     plt.figure()
        #     for k in range(76):
        #         x = obj_xy[k, :, 0]
        #         plt.plot(x, rms_wfe[k, j, :], color=colors[k])
        #         plt.scatter(x, rms_wfe[k, j, :], color=colors[k], s=10)
        #         plt.xlabel('Along Slice')
        # plt.show()

        rms_maps.append(rms_wfe)
        focal_coord.append(foc_xy)
        object_coord.append(obj_xy)

    # Stitch the different IFU sections
    rms_field = np.concatenate(rms_maps, axis=1)

    min_rms = np.min(rms_field)
    max_rms = np.max(rms_field)

    # (1) DETECTOR plane plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # Loop over the IFU channels: AB, CD, EF, GH
    # for i in range(2):
    i = 0
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
        title = r'IFU-%s | %s | %s | %s' % (ifu_section, spaxel_scale, grating, ao_mode)
        ax.set_title(title)

    fig_name = "RMSWFE_%s_DETECTOR_SPEC_%s_MODE_%s" % (spaxel_scale, grating, ao_mode)

    save_path = os.path.join(results_path, analysis_dir)
    if os.path.isfile(os.path.join(save_path, fig_name)):
        os.remove(os.path.join(save_path, fig_name))
    fig.savefig(os.path.join(save_path, fig_name))

    N_waves = len(waves)
    rms_maps = np.array(rms_maps)
    object_coord = np.array(object_coord)

    # (2) Stitch the Object space coordinates
    min_rms, max_rms = np.min(rms_maps), np.max(rms_maps)
    for k_wave, wave_str in zip([0, N_waves // 2, -1], ['MIN', 'CENT', 'MAX']):
        fig_obj, ax = plt.subplots(1, 1)
        x_obj, y_obj = object_coord[:, :, :, 0].flatten(), object_coord[:, :, :, 1].flatten()

        # we have to transform the object coordinates [mm] into arcseconds using the plate scales
        plate_scale = plates[spaxel_scale]
        x_obj /= plate_scale
        y_obj /= plate_scale

        triang = tri.Triangulation(x_obj, y_obj)
        rms_ = rms_maps[:, :, k_wave].flatten()
        tpc = ax.tripcolor(triang, rms_, shading='flat', cmap='jet')

        tpc.set_clim(vmin=min_rms, vmax=max_rms)
        # ax.scatter(x_obj, y_obj, s=3, color='black')

        axis_label = 'Object'
        ax.set_xlabel(axis_label + r' X [arcsec]')
        ax.set_ylabel(axis_label + r' Y [arcsec]')
        # ax.set_aspect('equal')
        ax.set_aspect(4)
        # cbar = plt.colorbar(tpc, ax=ax, orientation='horizontal')
        cbar = plt.colorbar(tpc, ax=ax)
        cbar.ax.set_xlabel('[nm]')

        wave = waves[k_wave]

        title = r'%s mas | %s SPEC %.3f $\mu$m | %s' % (spaxel_scale, grating, wave, ao_mode)
        ax.set_title(title)
        fig_name = "RMSWFE_OBJECT_%s_SPEC_%s_MODE_%s_WAVE_%s" % (spaxel_scale, grating, ao_mode, wave_str)

        save_path = os.path.join(results_path, analysis_dir)
        if os.path.isfile(os.path.join(save_path, fig_name)):
            os.remove(os.path.join(save_path, fig_name))
        fig_obj.savefig(os.path.join(save_path, fig_name))

    return rms_field


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Plate Scales in mm / arcsec
    plates = {'4x4': 1.0,
              '10x10': 1.0,
              '20x20': 1.0,
              '60x30': 1.0}

    # Nominal requirements for RMS WFE
    RMS_WFE = {'4x4': 87, '10x10': 134, '20x20': 259, '60x30': 553}

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    # [*] Monte Carlo Instances [*]
    ao_mode = 'NOAO'
    spaxel_scale = '60x30'
    spaxels_per_slice = 10       # How many field points per Slice to use
    pupil_sampling = 4          # N x N grid per pupil quadrant. See Zemax Operand help for RWRE
    gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_SHORT', 'K_LONG']
    file_options = {'MONTE_CARLO': True, 'AO_MODE': ao_mode, 'SPAX_SCALE': spaxel_scale, 'SLICE_SAMPLING': spaxels_per_slice,
                    'FPRS_MC': '0694', 'IPO_MC': '0055', 'IFU_PATHS_MC': {'AB': '0028', 'CD': '0007'}, 'ISP_MC': '0024'}

    files_path = os.path.abspath("D:/End to End Model/Monte_Carlo_Dec/Median")
    results_path = os.path.abspath("D:/End to End Model/Monte_Carlo_Dec/Median/Results/Mode_%s/Scale_%s" % (ao_mode, spaxel_scale))
    # [*] Monte Carlo Instances [*]

    # # [*] This is the bit we have to change when you run the analysis in your system [*]
    # sys_mode = 'HARMONI'
    # ao_modes = ['NOAO']
    # spaxel_scale = '4x4'
    # spaxels_per_slice = 3       # How many field points per Slice to use
    # pupil_sampling = 4          # N x N grid per pupil quadrant. See Zemax Operand help for RWRE
    # # gratings = ['VIS', 'Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K_SHORT', 'K_LONG']
    # gratings = ['H']
    #
    # file_options = {'MONTE_CARLO': False, 'which_system': sys_mode, 'AO_modes': ao_modes,
    #                 'SPAX_SCALE': spaxel_scale, 'AO_MODE': ao_modes[0]}
    #
    # files_path = os.path.abspath("D:/End to End Model/August_2020")
    # results_path = os.path.abspath("D:/End to End Model/Results_ReportAugust/Mode_%s/Scale_%s" % (ao_modes[0], spaxel_scale))
    # # [*] This is the bit we have to change when you run the analysis in your system [*]

    analysis_dir = os.path.join(results_path, 'RMS_WFE')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    # (1) First of all, we save a txt file with the MC Instances used to generate the E2E files (if running MC mode)
    # for each of the subsystems, as well as the slice sampling (field points per slice)
    if file_options['MONTE_CARLO'] is True:
        with open(os.path.join(analysis_dir, 'MC_settings.txt'), 'w') as file:
            file.write(json.dumps(file_options))

    # (2) We loop through the list of gratings we will use in this analysis
    rms_grating = []
    for grating in gratings:
        # change the grating option
        file_options['GRATING'] = grating
        rms = detector_rms_wfe(zosapi=psa, file_options=file_options, spaxels_per_slice=spaxels_per_slice, pupil_sampling=pupil_sampling,
                               files_path=files_path, results_path=results_path)
        rms_grating.append(rms.flatten())

    rms_grating = np.array(rms_grating).T




    # # First we want to justify the choice of Pupil Sampling.
    # # Comment this out if you want to run the normal analysis
    # # for grating in gratings:
    # #     pupil_sampling_effect(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
    # #                           spaxels_per_slice=spaxels_per_slice, grating=grating, files_path=files_path, results_path=results_path)
    #
    # # This is the core of the analysis. We loop over all spectral bands, calculating the RMS WFE
    # rms_grating = []
    # for grating in gratings:
    #     rms = detector_rms_wfe(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
    #                            spaxels_per_slice=spaxels_per_slice, grating=grating, pupil_sampling=pupil_sampling,
    #                            files_path=files_path, results_path=results_path)
    #     rms_grating.append(rms.flatten())
    #
    # rms_grating = np.array(rms_grating).T

    # We will save the results in a .txt file
    file_name = 'RMS_WFE_%s_%s_Percentiles.txt' % (ao_mode, spaxel_scale)
    with open(os.path.join(analysis_dir, file_name), 'w') as f:
        f.write('AO: %s, Spaxel Scale: %s\n' % (ao_mode, spaxel_scale))
        f.write('Field Points per slice: %d\n' % spaxels_per_slice)
        f.write('Gratings: ')
        f.write(str(gratings))
        f.write('\nRMS WFE Percentiles:\n')
        for k in range(len(gratings)):
            grating = gratings[k]
            rms_data = rms_grating[:, k]
            low_pctile = np.percentile(rms_data, 5)
            high_pctile = np.percentile(rms_data, 95)
            f.write("\n%s band: 5th = %.2f nm | 95th = %.2f nm" % (grating, low_pctile, high_pctile))

    data = pd.DataFrame(rms_grating, columns=gratings)

    # Violinplot
    fig_violin, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.violinplot(data=data, ax=ax, hue_order=gratings, palette=sns.color_palette("Reds"))
    ax.set_ylabel(r'RMS WFE [nm]')
    ax.set_title(r'RMS WFE Detector | %s scale | %s' % (spaxel_scale, ao_mode))
    # We add a line showing the nominal requirement, to see how far away we are
    ax.axhline(y=RMS_WFE[spaxel_scale], linestyle='-.', color='red')
    ax.set_ylim([0, 1.15*RMS_WFE[spaxel_scale]])
    # ax.set_ylim(bottom=0)

    fig_name = "RMS_WFE_DETECTOR_%s_%s_violin" % (spaxel_scale, ao_mode)
    analysis_dir = os.path.join(results_path, 'RMS_WFE')
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig_violin.savefig(os.path.join(analysis_dir, fig_name))

    # Box and Whisker plot across all spectral bands
    fig_box, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.boxplot(data=data, ax=ax, hue_order=gratings, palette=sns.color_palette("Reds"))
    ax.set_ylabel(r'RMS WFE [nm]')
    ax.set_title(r'RMS WFE Detector | %s scale | %s' % (spaxel_scale, ao_mode))
    ax.axhline(y=RMS_WFE[spaxel_scale], linestyle='-.', color='red')
    ax.set_ylim([0, 1.15*RMS_WFE[spaxel_scale]])
    # ax.set_ylim(bottom=0)

    fig_name = "RMS_WFE_DETECTOR_%s_%s" % (spaxel_scale, ao_mode)
    analysis_dir = os.path.join(results_path, 'RMS_WFE')
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig_box.savefig(os.path.join(analysis_dir, fig_name))

    plt.show()
