"""
RMS Wavefront Error analysis


Author: Alvaro Menduina
Date: April 2020

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


def rms_wfe_histograms(zosapi, mode, spaxel_scale, spaxels_per_slice, wavelength_idx, spectral_band, plane,
                       files_path, results_path):

    """

    :param zosapi:
    :param spaxel_scale:
    :param spaxels_per_slice:
    :param wavelength_idx:
    :param spectral_band:
    :param plane:
    :param files_path:
    :param results_path:
    :return:
    """

    if mode == 'IFS':
        ao_modes = []
    elif mode == 'HARMONI':
        ao_modes = ['NOAO']

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    analysis = e2e.RMS_WFE_Analysis(zosapi=zosapi)

    rms = []
    for ifu_section in ifu_sections:

        options = {'which_system': mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [spectral_band]}
        # We need to know the Surface Number for the focal plane of interest in the Zemax files
        # this is something varies with mode, scale, ifu channel so we save those numbers on e2e.focal_planes dictionary
        focal_plane = e2e.focal_planes[mode][spaxel_scale][ifu_section][plane]
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=None,
                                                surface=focal_plane, spaxels_per_slice=spaxels_per_slice)
        # Only 1 list, no Monte Carlo
        rms_wfe, obj_xy, foc_xy, global_xy, waves = list_results[0]
        rms.append(rms_wfe)

    max_rms = np.round(np.max(rms)) + 1
    min_rms = np.round(np.min(rms)) - 2

    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        ax = axes.flatten()[i]
        ax.hist(rms[i].flatten(), bins=np.arange(0, max_rms, 2), histtype='step', color='red')
        ax.set_title(r'IFU-%s' % ifu_sections[i])
        ax.set_xlabel(r'RMS WFE [nm]')
        ax.set_ylabel(r'Frequency')
        ax.set_xlim([min_rms, max_rms])
        ax.set_xticks(np.arange(0, max_rms, 5))

    fig_name = "HISTOGRAM_RMS_%s_SPEC_%s_SURF_%s" % (spaxel_scale, spectral_band, plane)
    if os.path.isfile(os.path.join(results_path, fig_name)):
        os.remove(os.path.join(results_path, fig_name))
    fig.savefig(os.path.join(results_path, fig_name))

    return


def stitch_fields(zosapi, mode, spaxel_scale, spaxels_per_slice, wavelength_idx, spectral_band, plane,
                  files_path, results_path, show='focal'):
    """
    Loop over the IFU sections AB, CD, EF, GH and calculate the RMS WFE map for each case
    Gather the results and stitch them together to get a complete Field Map of the IFU

    We can specify the Focal Plane at which we calculate the RMS. For example:
    'PO': preoptics, 'IS': Image Slicer, 'DET': detector

    We have to be careful with the plot format because the coordinate systems vary depending on
    the focal plane we choose. Moreover, we can choose whether to refer the results to 'object' or
    'focal' coordinates, which also depends on the surface. For instance, if we go to the DETECTOR plane,
    it makes no sense to use 'object' coordinates because the wavelengths would overlap, we must use 'focal'

    :param zosapi: the Python Standalone Application
    :param mode: E2E file mode: IFS, HARMONI or ELT
    :param spaxel_scale: string, spaxel scale like '60x30'
    :param spaxels_per_slice: how many points to sample each slice.
    :param wavelength_idx: list of Zemax wavelength indices to analyse
    :param spectral_band: string, the grating we use
    :param plane: string, codename for the focal plane surface 'PO': PreOptics, 'IS': Image Slicer, 'DET': Detector
    :param files_path, path where the E2E model Zemax files are stored
    :param results_path, path where we want to save the plots
    :param show: string, either 'focal' for focal plane coordinates or 'object' for object space coordinates
    :return:
    """

    if mode == 'IFS':
        ao_modes = []
    elif mode == 'HARMONI':
        ao_modes = ['NOAO']

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    analysis = e2e.RMS_WFE_Analysis(zosapi=zosapi)
    rms_maps, object_coord, focal_coord = [], [], []

    for ifu_section in ifu_sections:

        options = {'which_system': mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [spectral_band]}
        # We need to know the Surface Number for the focal plane of interest in the Zemax files
        # this is something varies with mode, scale, ifu channel so we save those numbers on e2e.focal_planes dictionary
        focal_plane = e2e.focal_planes[mode][spaxel_scale][ifu_section][plane]
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=None,
                                                surface=focal_plane, spaxels_per_slice=spaxels_per_slice)
        # Only 1 list, no Monte Carlo
        rms_wfe, obj_xy, foc_xy, global_xy, waves = list_results[0]

        rms_maps.append(rms_wfe)
        focal_coord.append(foc_xy)
        object_coord.append(obj_xy)

    # Stitch the different IFU sections
    rms_field = np.concatenate(rms_maps, axis=1)
    obj_xy = np.concatenate(object_coord, axis=1)
    foc_xy = np.concatenate(focal_coord, axis=1)

    if plane != 'DET':

        # For all surfaces except the detector plane we want to separate the plots for each wavelength

        min_rms = np.min(rms_field)
        max_rms = np.max(rms_field)

        for j_wave, wavelength in enumerate(waves):

            if show == 'object':
                x, y = obj_xy[j_wave, :, :, 0].flatten(), obj_xy[j_wave, :, :, 1].flatten()
                axis_label = 'Object Plane'
                ref = 'OBJ'

            elif show == 'focal':
                x, y = foc_xy[j_wave, :, :, 0].flatten(), foc_xy[j_wave, :, :, 1].flatten()
                axis_label = 'Focal Plane'
                ref = 'FOC'

            if show == 'object' or show == 'focal':

                triang = tri.Triangulation(x, y)

                fig, ax = plt.subplots(1, 1)
                ax.set_aspect('equal')
                tpc = ax.tripcolor(triang, rms_field[j_wave, :, :].flatten(), shading='flat', cmap='jet')
                tpc.set_clim(vmin=min_rms, vmax=max_rms)
                # ax.scatter(x, y, s=2, color='black')
                ax.set_xlabel(axis_label + r' X [mm]')
                ax.set_ylabel(axis_label + r' Y [mm]')
                title = r'RMS Field Map Stitched IFU | %s mas %s %.3f $\mu$m' % (spaxel_scale, spectral_band, wavelength)
                ax.set_title(title)
                plt.colorbar(tpc, ax=ax)
                fig_name = "STITCHED_RMSMAP_%s_SPEC_%s_%s_SURF_%s_WAVE%d" % (
                spaxel_scale, spectral_band, ref, plane, wavelength_idx[j_wave])
                if os.path.isfile(os.path.join(results_path, fig_name)):
                    os.remove(os.path.join(results_path, fig_name))
                fig.savefig(os.path.join(results_path, fig_name))

            elif show == 'both':

                x_obj, y_obj = obj_xy[j_wave, :, :, 0].flatten(), obj_xy[j_wave, :, :, 1].flatten()
                x_foc, y_foc = foc_xy[j_wave, :, :, 0].flatten(), foc_xy[j_wave, :, :, 1].flatten()
                xy = [[x_obj, y_obj], [x_foc, y_foc]]
                axis_labels = ['Object Plane', 'Focal Plane']
                refs = ['OBJ', 'FOC']

                for i in range(2):
                    x, y = xy[i]
                    triang = tri.Triangulation(x, y)
                    fig, ax = plt.subplots(1, 1)
                    ax.set_aspect('equal')
                    tpc = ax.tripcolor(triang, rms_field[j_wave, :, :].flatten(), shading='flat', cmap='jet')
                    tpc.set_clim(vmin=min_rms, vmax=max_rms)
                    ax.set_xlabel(axis_labels[i] + r' X [mm]')
                    ax.set_ylabel(axis_labels[i] + r' Y [mm]')
                    title = r'RMS Field Map Stitched IFU | %s mas %s %.3f $\mu$m' % (
                    spaxel_scale, spectral_band, wavelength)
                    ax.set_title(title)
                    plt.colorbar(tpc, ax=ax)
                    fig_name = "STITCHED_RMSMAP_%s_SPEC_%s_%s_SURF_%s_WAVE%d" % (
                        spaxel_scale, spectral_band, refs[i], plane, wavelength_idx[j_wave])
                    if os.path.isfile(os.path.join(results_path, fig_name)):
                        os.remove(os.path.join(results_path, fig_name))
                    fig.savefig(os.path.join(results_path, fig_name))

            else:
                raise ValueError('show should be "object" / "focal" / "both"')

    if plane == 'DET':
        # For the detector plane we want to plot ALL wavelengths at the same time!
        # And using the FOCAL PLANE COORDINATES as reference
        # at the Object plane there is an overlap in wavelength!!

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

                x_odd, y_odd = _foc_xy[:, ::2, :, 0].flatten(), _foc_xy[:, ::2, :, 1].flatten()
                x_even, y_even = _foc_xy[:, 1::2, :, 0].flatten(), _foc_xy[:, 1::2, :, 1].flatten()
                triang_odd = tri.Triangulation(x_odd, y_odd)
                triang_even = tri.Triangulation(x_even, y_even)

                # Remove the flat triangles at the detector edges that appear because of the field curvature
                min_circle_ratio = .05
                mask_odd = tri.TriAnalyzer(triang_odd).get_flat_tri_mask(min_circle_ratio)
                triang_odd.set_mask(mask_odd)
                mask_even = tri.TriAnalyzer(triang_even).get_flat_tri_mask(min_circle_ratio)
                triang_even.set_mask(mask_even)

                tpc_odd = ax.tripcolor(triang_odd, _rms_field[:, ::2].flatten(), shading='flat', cmap='jet')
                tpc_odd.set_clim(vmin=min_rms, vmax=max_rms)
                tpc_even = ax.tripcolor(triang_even, _rms_field[:, 1::2].flatten(), shading='flat', cmap='jet')
                tpc_even.set_clim(vmin=min_rms, vmax=max_rms)
                # ax.scatter(x, y, s=2, color='black')

                draw_detector_boundary(ax)

                axis_label = 'Detector'
                ax.set_xlabel(axis_label + r' X [mm]')
                ax.set_ylabel(axis_label + r' Y [mm]')
                ax.set_aspect('equal')
                cbar = plt.colorbar(tpc_odd, ax=ax, orientation='horizontal')
                cbar.ax.set_xlabel('[nm]')
                title = r'IFU-%s | %s mas | %s SPEC | %s' % (ifu_section, spaxel_scale, spectral_band, mode)
                ax.set_title(title)
                fig_name = "RMSMAP_%s_DETECTOR_SPEC_%s_MODE_%s" % (spaxel_scale, spectral_band, mode)
                if os.path.isfile(os.path.join(results_path, fig_name)):
                    os.remove(os.path.join(results_path, fig_name))
                fig.savefig(os.path.join(results_path, fig_name))
                # plt.close(fig)

        # # Plot the histograms
        # fig_hist, axes = plt.subplots(2, 2, figsize=(10, 10))
        # for i in range(4):
        #     ax = axes.flatten()[i]
        #     ax.hist(rms_maps[i].flatten(), bins=np.arange(0, max_rms, 2), histtype='step', color='red')
        #     ax.set_title(r'IFU-%s' % ifu_sections[i])
        #     ax.set_xlabel(r'RMS WFE [nm]')
        #     ax.set_ylabel(r'Frequency')
        #     ax.set_xlim([min_rms, max_rms])
        #     ax.set_xticks(np.arange(0, max_rms, 5))
        #
        # fig_name = "RMSMAP_%s_DETECTOR_SPEC_%s_HISTOGRAM" % (spaxel_scale, spectral_band)
        # if os.path.isfile(os.path.join(results_path, fig_name)):
        #     os.remove(os.path.join(results_path, fig_name))
        # fig_hist.savefig(os.path.join(results_path, fig_name))
        # plt.close(fig_hist)

        # Box and Whisker plots

        rms_flats = [array.flatten() for array in rms_maps]
        rms_flats = np.array(rms_flats).T

        data = pd.DataFrame(rms_flats, columns=ifu_sections)

        fig_box, ax = plt.subplots(1, 1)
        sns.boxplot(data=data, ax=ax)
        ax.set_ylabel(r'RMS WFE [nm]')
        ax.set_title(r'RMS WFE Detector | %s scale | %s band | %s' % (spaxel_scale, spectral_band, mode))
        fig_name = "RMSMAP_%s_DETECTOR_SPEC_%s_BOXWHISKER_MODE_%s" % (spaxel_scale, spectral_band, mode)
        if os.path.isfile(os.path.join(results_path, fig_name)):
            os.remove(os.path.join(results_path, fig_name))
        fig_box.savefig(os.path.join(results_path, fig_name))
        plt.close(fig_box)


    return rms_field, obj_xy, foc_xy, rms_maps, object_coord, focal_coord


def image_slicer_analysis(zosapi, spaxel_scale, spaxels_per_slice, wavelength_idx, grating, files_path, results_path):
    """

    The Stitching does not work at the surface BEFORE the Image Slicer if you use the Focal Plane Coordinates
    because the Odd/Even configurations overlap

    For that reason, we need to treat this surface differently. We have to split the results between
    Odd / Even configurations (i.e. the A and B paths) and plot them alongside, rather than stitching

    :param zosapi: the Python Standalone Application
    :param spaxel_scale: string, spaxel scale like '60x30'
    :param spaxels_per_slice: how many points to sample each slice.
    :param wavelength_idx: list of Zemax wavelength indices to analyse
    :param spectral_band: string, the grating we use
    :param files_path, path where the E2E model Zemax files are stored
    :param results_path, path where we want to save the plots
    :return:
    """

    # Based on the Field Definition from HRM-00261 and looking at the E2E Zemax files
    # to see which Configurations have Positive or Negative field values.
    ifu_configs = {'AB': {'Odd': 'A', 'Even': 'B'},
                   'CD': {'Even': 'C', 'Odd': 'D'},
                   'EF': {'Odd': 'E', 'Even': 'F'},
                   'GH': {'Even': 'G', 'Odd': 'H'}}

    analysis = e2e.RMS_WFE_Analysis(zosapi=zosapi)

    for ifu in ['AB', 'CD', 'EF', 'GH']:        # Loop over IFU sections

        options = {'which_system': 'IFS', 'AO_modes': [], 'scales': [spaxel_scale], 'IFUs': [ifu],
                   'grating': [grating]}
        focal_plane = e2e.focal_planes['IFS'][spaxel_scale][ifu]['IS']

        zemax_files, settings = e2e.create_zemax_file_list(which_system='IFS', scales=[spaxel_scale], IFUs=[ifu],
                                                 grating=[grating], AO_modes=[])
        zemax_file = zemax_files[0]
        print(zemax_file)
        file_name = zemax_file.split('.')[0]
        results_dir = os.path.join(results_path, file_name)

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options,
                                                                  results_path=results_path, wavelength_idx=wavelength_idx,
                                                                  configuration_idx=None, surface=focal_plane,
                                                                  spaxels_per_slice=spaxels_per_slice)

        rms_wfe, obj_xy, foc_xy, global_xy, waves = list_results[0]

        analysis_dir = os.path.join(results_dir, 'RMS_WFE')
        print("Analysis Results will be saved in folder: ", analysis_dir)
        if not os.path.exists(analysis_dir):
            os.mkdir(analysis_dir)

        fig, axes = plt.subplots(1, 2)
        configs = ['Even', 'Odd']
        colors = ['black', 'white']
        min_rms, max_rms = np.min(rms_wfe[0]), np.max(rms_wfe[0])
        for i in range(2):
            ax = axes[i]
            x, y = foc_xy[0, 1 - i::2, :, 0].flatten(), foc_xy[0, 1 - i::2, :, 1].flatten()
            triang = tri.Triangulation(x, y)
            tpc = ax.tripcolor(triang, rms_wfe[0, 1 - i::2, :].flatten(), shading='flat', cmap='jet')
            tpc.set_clim(vmin=min_rms, vmax=max_rms)
            ax.scatter(x, y, s=2, color=colors[i])
            ax.set_xlabel(r'Focal Plane X [mm]')
            if i == 0:
                ax.set_ylabel(r'Focal Plane Y [mm]')
            ax.set_aspect('equal')
            ax.set_title('IFU-%s | %s' % (ifu_configs[ifu][configs[i]], configs[i]))
            plt.colorbar(tpc, ax=ax, orientation='horizontal')

        fig_name = file_name + '_RMSWFE_IMAGE_SLICER_WAVE0'
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        fig.savefig(os.path.join(analysis_dir, fig_name))
        # plt.close(fig)

    return

def slit_analysis(zosapi, spaxel_scale, spaxels_per_slice, wavelength_idx, grating, files_path, results_path):
    """
    Due to the particular morphology of the IFU slits, we need to use custom plots for this surface
    We use scatter plots, where the color and size of the marker depends on the RMS WFE

    :param zosapi: the Python Standalone Application
    :param spaxel_scale: string, spaxel scale like '60x30'
    :param spaxels_per_slice: how many points to sample each slice.
    :param wavelength_idx: list of Zemax wavelength indices to analyse
    :param spectral_band: string, the grating we use
    :param files_path, path where the E2E model Zemax files are stored
    :param results_path, path where we want to save the plots
    :return:
    """

    # Based on the Field Definition from HRM-00261 and looking at the E2E Zemax files
    # to see which Configurations have Positive or Negative field values.
    ifu_configs = {'AB': {'Odd': 'A', 'Even': 'B'},
                   'CD': {'Even': 'C', 'Odd': 'D'},
                   'EF': {'Odd': 'E', 'Even': 'F'},
                   'GH': {'Even': 'G', 'Odd': 'H'}}

    analysis = e2e.RMS_WFE_Analysis(zosapi=zosapi)

    for ifu in ['AB', 'CD', 'EF', 'GH']:

        options = {'which_system': 'IFS', 'AO_modes': [], 'scales': [spaxel_scale], 'IFUs': [ifu], 'grating': [grating]}
        slit_plane = e2e.focal_planes['IFS'][spaxel_scale][ifu]['SL']

        zemax_files, settings = e2e.create_zemax_file_list(which_system='IFS', scales=[spaxel_scale], IFUs=[ifu],
                                                 grating=[grating], AO_modes=[])
        zemax_file = zemax_files[0]
        print(zemax_file)
        file_name = zemax_file.split('.')[0]
        results_dir = os.path.join(results_path, file_name)

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options,
                                                          results_path=results_path, wavelength_idx=wavelength_idx,
                                                          configuration_idx=None, surface=slit_plane,
                                                          spaxels_per_slice=spaxels_per_slice)

        rms_slit, obj_xy, foc_xy, global_xy, waves = list_results[0]

        analysis_dir = os.path.join(results_dir, 'RMS_WFE')
        print("Analysis Results will be saved in folder: ", analysis_dir)
        if not os.path.exists(analysis_dir):
            os.mkdir(analysis_dir)

        i_wave = 0
        min_rms = np.min(rms_slit[i_wave])
        max_rms = np.max(rms_slit[i_wave])
        smin, smax = 5, 20
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for j, (label, marker) in enumerate(zip(['Odd', 'Even'], ['^', 'v'])):

            ax = axes[j]
            foc_x = foc_xy[i_wave, j::2, :, 0].flatten()
            foc_y = foc_xy[i_wave, j::2, :, 1].flatten()
            rms_values = rms_slit[i_wave, j::2, :].flatten()

            ratios = (rms_values - min_rms) / (max_rms - min_rms)
            s = smin + (smax - smin) * ratios

            channel = ifu_configs[ifu][label]
            scat = ax.scatter(foc_x, foc_y, s=s, label='IFU-' + channel, marker=marker, c=rms_values, cmap='jet')
            scat.set_clim(vmin=min_rms, vmax=max_rms)
            plt.colorbar(scat, ax=ax, orientation='horizontal')
            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Y [mm]')
            ax.legend()

        plt.tight_layout()

        fig_name = file_name + '_RMSWFE_SLITS_WAVE1'
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        fig.savefig(os.path.join(analysis_dir, fig_name))
        # plt.close(fig)

    return


if __name__ == """__main__""":


    # grating = 'H'

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\June_2020")
    results_path = os.path.abspath("D:\End to End Model\Results_June")

    # analysis = e2e.RMS_WFE_Analysis(zosapi=psa)
    # options = {'which_system': 'IFS', 'AO_modes': [], 'scales': ['60x30'], 'IFUs': ['AB'],
    #            'grating': ['H']}
    # # We need to know the Surface Number for the focal plane of interest in the Zemax files
    # # this is something varies with mode, scale, ifu channel so we save those numbers on e2e.focal_planes dictionary
    # focal_plane = e2e.focal_planes['IFS']['60x30']['AB']['PO']
    # list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
    #                                         wavelength_idx=None, configuration_idx=None,
    #                                         surface=focal_plane, spaxels_per_slice=15)
    #
    # import h5py
    # path_hdf5 = os.path.join(results_path, 'HDF5')
    # path_to_file = os.path.join(path_hdf5, 'CRYO_PO60x30_IFUAB_SPECH.hdf5')
    #
    #
    # zemax_metadata = e2e.read_hdf5(path_to_file)

    " (5) Detector Plane "

    # Things to do:

    # Get RMS WFE maps / histograms / box & whiskers for ALL Scales and ALL Gratings

    analysis_dir = os.path.join(results_path, 'RMS_WFE')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    gratings = ['Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    mode = 'HARMONI'
    for spaxel_scale in ['4x4']:

        rms_grating = []
        for grating in gratings:
            stitched_results = stitch_fields(zosapi=psa, mode=mode, spaxel_scale=spaxel_scale, spaxels_per_slice=3, wavelength_idx=None,
                                             spectral_band=grating, plane='DET', files_path=files_path, results_path=results_path,
                                             show='focal')
            rms_allifu = stitched_results[0]
            rms_grating.append(rms_allifu.flatten())

        rms_grating = np.array(rms_grating).T

        data = pd.DataFrame(rms_grating, columns=gratings)

        fig_box, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.boxplot(data=data, ax=ax, hue_order=gratings, palette=sns.color_palette("Blues"))
        ax.set_ylabel(r'RMS WFE [nm]')
        ax.set_title(r'RMS WFE Detector | %s scale | %s' % (spaxel_scale, mode))

        fig_name = "RMSMAP_%s_DETECTOR_ALL_SPEC_BOXWHISKER_%s" % (spaxel_scale, mode)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        fig_box.savefig(os.path.join(analysis_dir, fig_name))

    plt.show()




    rms_wfe_histograms(zosapi=psa, mode=mode, spaxel_scale=spaxel_scale, spaxels_per_slice=3, wavelength_idx=None,
                                     spectral_band=grating, plane='DET', files_path=files_path, results_path=results_path)

    # Fix the 60x30 surface dictionary

    " (1) PreOptics focal plane"
    # We start by showing the RMS WFE map after stitching all IFU channels
    # at the PreOptics focal plane
    # both in Focal Plane coordinates and Object Space coordinates
    # for the chosen grating at 3 wavelengths (shortest, central, longest)
    stitched_results = stitch_fields(zosapi=psa, mode=mode, spaxel_scale=spaxel_scale, spaxels_per_slice=10, wavelength_idx=[1, 12, 23],
                                     spectral_band=grating, plane='PO', files_path=files_path, results_path=results_path,
                                     show='both')
    plt.show()

    " (2) Pre Image Slicer analysis "
    # Here, we show why Stitching together the different IFU sections does not work for some surfaces
    # Note: This only happens in FOCAL Plane Coordinates!
    # Right before the Image Slicer, the focal plane coordinates for the Odd and Even configurations overlap
    # This ruins any attempt at plotting the results for A and B sections together.

    # To demonstrate this, we plot both Object and Focal coordinates for Odd/Even configurations
    # at 2 separate surfaces: (1) exit of the PreOptics, (2) at the Image Slicer

    analysis = e2e.RMS_WFE_Analysis(zosapi=psa)
    ifu = 'AB'
    wavelength_idx = [1]
    spaxels_per_slice = 5
    options = {'which_system': 'IFS', 'AO_modes': [], 'scales': [spaxel_scale], 'IFUs': [ifu], 'grating': [grating]}
    zemax_files, settings = e2e.create_zemax_file_list(which_system='IFS', scales=[spaxel_scale], IFUs=[ifu],
                                                       grating=[grating], AO_modes=[])
    zemax_file = zemax_files[0]
    file_name = zemax_file.split('.')[0]
    results_dir = os.path.join(results_path, file_name)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    analysis_dir = os.path.join(results_dir, 'RMS_WFE')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    surfaces = ['PO', 'IS']
    surface_names = ['PreOptics', 'Image Slicer']

    s = 10
    fig, axes = plt.subplots(len(surfaces), 2, figsize=(9, 9))
    for i, (surf, name) in enumerate(zip(surfaces, surface_names)):

        focal_plane = e2e.focal_planes['IFS'][spaxel_scale][ifu][surf]
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options,
                                                                  results_path=results_path, wavelength_idx=wavelength_idx,
                                                                  configuration_idx=None, surface=focal_plane,
                                                                  spaxels_per_slice=spaxels_per_slice)

        rms, obj_xy, foc_xy, global_xy, waves = list_results[0]

        for j, (ref, label) in enumerate(zip([obj_xy, foc_xy], ['Object', 'Focal'])):
            ax = axes[i][j]
            ax.scatter(ref[0, 0::2, :, 0], ref[0, 0::2, :, 1], color='blue', s=s, marker='v', alpha=0.5, label='Odd')
            ax.scatter(ref[0, 1::2, :, 0], ref[0, 1::2, :, 1], color='red', s=s, marker='^', alpha=0.5, label='Even')
            ax.legend(title='Configs', loc=2)
            if i == len(surfaces) - 1:
                ax.set_xlabel(label + r' X [mm]')
            ax.set_ylabel(label + r' Y [mm]')
            ax.set_title(r'IFU-%s | Surface: %s' % (ifu, name))

    fig_name = file_name + '_COORDINATES'
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig.savefig(os.path.join(analysis_dir, fig_name))
    plt.show()

    " (3) Image Slicer "
    # Show the RMS WFE for each IFU channel at the Image Slicer
    image_slicer_analysis(zosapi=psa, spaxel_scale=spaxel_scale, spaxels_per_slice=15, wavelength_idx=[1], grating=grating,
                          files_path=files_path, results_path=results_path)
    plt.show()

    " (4) IFU Slits "

    slit_analysis(zosapi=psa, spaxel_scale=spaxel_scale, spaxels_per_slice=5, wavelength_idx=[1], grating=grating,
                  files_path=files_path, results_path=results_path)
    plt.show()




