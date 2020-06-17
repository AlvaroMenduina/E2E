"""

RMS Wavefront Error analysis


Author: Alvaro Menduina
Date: June 2020

Description:
With this script we calculate the RMS Wavefront Error at the detector plane
for the E2E files

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

def detector_rms_wfe(zosapi, sys_mode, ao_modes, spaxel_scale, spaxels_per_slice, grating, files_path, results_path):


    analysis_dir = os.path.join(results_path, 'RMS_WFE')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']
    analysis = e2e.RMS_WFE_Analysis(zosapi=zosapi)

    rms_maps, object_coord, focal_coord = [], [], []

    for ifu_section in ifu_sections:

        options = {'which_system': sys_mode, 'AO_modes': ao_modes, 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}

        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=None, configuration_idx=None,
                                                surface=None, spaxels_per_slice=spaxels_per_slice)
        # Only 1 list, no Monte Carlo
        rms_wfe, obj_xy, foc_xy, global_xy, waves = list_results[0]

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

            # Separate Odd and Even configs to avoid triangulating over the detector gap
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
            title = r'IFU-%s | %s mas | %s SPEC | %s' % (ifu_section, spaxel_scale, grating, sys_mode)
            ax.set_title(title)
            fig_name = "RMSMAP_%s_DETECTOR_SPEC_%s_MODE_%s" % (spaxel_scale, grating, sys_mode)

            save_path = os.path.join(results_path, analysis_dir)
            if os.path.isfile(os.path.join(save_path, fig_name)):
                os.remove(os.path.join(save_path, fig_name))
            fig.savefig(os.path.join(save_path, fig_name))

    return rms_field


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\June_2020")
    results_path = os.path.abspath("D:\End to End Model\Results_June")

    sys_mode = 'HARMONI'
    ao_modes = ['NOAO']
    spaxel_scale = '10x10'
    # gratings = ['Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    gratings = ['H', 'H_HIGH']
    analysis_dir = os.path.join(results_path, 'RMS_WFE')

    rms_grating = []
    for grating in gratings:
        rms = detector_rms_wfe(zosapi=psa, sys_mode=sys_mode, ao_modes=ao_modes, spaxel_scale=spaxel_scale,
                               spaxels_per_slice=3, grating=grating, files_path=files_path, results_path=results_path)
        rms_grating.append(rms.flatten())

    rms_grating = np.array(rms_grating).T

    data = pd.DataFrame(rms_grating, columns=gratings)

    fig_box, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.boxplot(data=data, ax=ax, hue_order=gratings, palette=sns.color_palette("Blues"))
    ax.set_ylabel(r'RMS WFE [nm]')
    ax.set_title(r'RMS WFE Detector | %s scale | %s' % (spaxel_scale, sys_mode))

    fig_name = "RMS_WFE_DETECTOR_%s_%s" % (spaxel_scale, sys_mode)
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig_box.savefig(os.path.join(analysis_dir, fig_name))

    plt.show()