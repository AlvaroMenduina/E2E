""""

"""

import os
import numpy as np
import e2e_analysis as e2e
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
import pandas as pd
import seaborn as sns


def stich_fwhm_detector(zosapi, spaxel_scale, grating):


    analysis = e2e.GeometricFWHM_PSF_Analysis(zosapi=zosapi)

    ifu_sections = ['AB', 'CD', 'EF', 'GH']

    analysis_dir = os.path.join(results_path, 'FWHM')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    fx, fy = [], []
    focal_coord = []
    fx_waves, fy_waves = [], []
    for ifu_section in ifu_sections:
        options = {'which_system': 'IFS', 'AO_modes': [], 'scales': [spaxel_scale], 'IFUs': [ifu_section],
                   'grating': [grating]}
        focal_plane = e2e.focal_planes['IFS'][spaxel_scale][ifu_section]['DET']

        wavelength_idx = np.linspace(1, 23, 5).astype(int)
        list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                                wavelength_idx=wavelength_idx, configuration_idx=None, surface=focal_plane,
                                                N_rays=1000)

        fwhm, psf_cube, obj_xy, foc_xy, wavelengths = list_results
        print("Max FWHM: %.1f" % np.nanmax(fwhm))
        focal_coord.append(foc_xy)

        fx_waves.append(fwhm[:, :, 0])
        fy_waves.append(fwhm[:, :, 1])

        fwhm_x, fwhm_y = fwhm[:, :, 0].flatten(), fwhm[:, :, 1].flatten()
        fx.append(fwhm_x)
        fy.append(fwhm_y)

    # for data in [fx_waves, fy_waves]:
    #     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    #     # Loop over the IFU channels: AB, CD, EF, GH
    #     for i in range(2):
    #         for j in range(2):
    #             k = 2 * i + j
    #             ifu_section = ifu_sections[k]
    #             ax = axes[i][j]
    #             _foc_xy = focal_coord[k]
    #             # _rms_field = rms_maps[k]
    #
    #             x_odd, y_odd = _foc_xy[:, ::2, :, 0].flatten(), _foc_xy[:, ::2, :, 1].flatten()
    #             x_even, y_even = _foc_xy[:, 1::2, :, 0].flatten(), _foc_xy[:, 1::2, :, 1].flatten()
    #             triang_odd = tri.Triangulation(x_odd, y_odd)
    #             triang_even = tri.Triangulation(x_even, y_even)
    #
    #             # Remove the flat triangles at the detector edges that appear because of the field curvature
    #             min_circle_ratio = .05
    #             mask_odd = tri.TriAnalyzer(triang_odd).get_flat_tri_mask(min_circle_ratio)
    #             triang_odd.set_mask(mask_odd)
    #             mask_even = tri.TriAnalyzer(triang_even).get_flat_tri_mask(min_circle_ratio)
    #             triang_even.set_mask(mask_even)
    #
    #             min_fwhm, max_fwhm = np.nanmin(data[k]), np.nanmax(data[k])
    #
    #             tpc_odd = ax.tripcolor(triang_odd, data[k][:, ::2].flatten(), shading='flat', cmap='jet')
    #             tpc_odd.set_clim(vmin=min_fwhm, vmax=max_fwhm)
    #             tpc_even = ax.tripcolor(triang_even, data[k][:, 1::2].flatten(), shading='flat', cmap='jet')
    #             tpc_even.set_clim(vmin=min_fwhm, vmax=max_fwhm)
    #             # ax.scatter(x, y, s=2, color='black')
    #             axis_label = 'Detector'
    #             ax.set_xlabel(axis_label + r' X [mm]')
    #             ax.set_ylabel(axis_label + r' Y [mm]')
    #             ax.set_aspect('equal')
    #             plt.colorbar(tpc_odd, ax=ax, orientation='horizontal')
    #
    # # plt.show()


    max_fwhm = max(np.nanmax(fx), np.nanmax(fy))
    print(max_fwhm)
    max_fwhm = np.round(max_fwhm) + 1
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, (fwhm_x, fwhm_y) in enumerate(zip(fx, fy)):
        ax = axes.flatten()[i]
        ax.hist(fwhm_x, bins=np.arange(0, max_fwhm), histtype='step', color='red', label='Along [X]')
        ax.hist(fwhm_y, bins=np.arange(0, max_fwhm), histtype='step', color='blue', label='Across [Y]')

        ax.set_title(r'IFU-%s | %s mas | %s SPEC' % (ifu_sections[i], spaxel_scale, grating))
        ax.set_xlabel(r'FWHM Geometric PSF [$\mu$m]')
        ax.set_ylabel(r'Frequency')
        ax.set_xlim([0, np.round(max_fwhm) + 1])
        ax.set_xticks(np.arange(0, max_fwhm))

        ax.legend()
    #
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()

    fig_name = 'FWHM_DET_' + spaxel_scale + '_SPEC_' + grating
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig.savefig(os.path.join(analysis_dir, fig_name))
    # plt.close(fig)

    plt.show()


    return fx_waves, fy_waves, focal_coord



if __name__ == """__main__""":


    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = e2e.PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\April_2020")
    results_path = os.path.abspath("D:\End to End Model\Results_April")

    analysis_dir = os.path.join(results_path, 'FWHM')
    print("Analysis Results will be saved in folder: ", analysis_dir)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    gratings = ['Z_HIGH', 'IZ', 'J', 'IZJ', 'H', 'H_HIGH', 'HK', 'K', 'K_LONG', 'K_SHORT']
    gratings = ['HK']

    spaxel_scale = '60x30'

    fx_grating, fy_grating = [], []
    for grating in gratings:
        fx_waves, fy_waves, focal_coord = stich_fwhm_detector(zosapi=psa, spaxel_scale=spaxel_scale, grating=grating)


        fx, fy, psf_cube = stich_fwhm_detector(zosapi=psa, spaxel_scale=spaxel_scale, grating=grating)
        fx_grating.append(np.array(fx).flatten())
        fy_grating.append(np.array(fy).flatten())

    fx_grating = np.array(fx_grating).T
    fy_grating = np.array(fy_grating).T


    data_x = pd.DataFrame(fx_grating, columns=gratings)
    data_y = pd.DataFrame(fy_grating, columns=gratings)

    max_val = np.round(max(np.max(fx_grating), np.max(fy_grating))) + 1

    fig_box, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    sns.boxplot(data=data_x, ax=ax1, hue_order=gratings, palette=sns.color_palette("Blues"))
    sns.boxplot(data=data_y, ax=ax2, hue_order=gratings, palette=sns.color_palette("Reds"))
    ax1.set_ylim([0, max_val])
    ax2.set_ylim([0, max_val])
    ax1.set_title('FWHM Along Slice [X]')
    ax2.set_title('FWHM Across Slice [Y]')

    fig_name = "FWHM_%s_DETECTOR_ALL_SPEC_BOXWHISKER" % (spaxel_scale)
    if os.path.isfile(os.path.join(analysis_dir, fig_name)):
        os.remove(os.path.join(analysis_dir, fig_name))
    fig_box.savefig(os.path.join(analysis_dir, fig_name))

    plt.show()

    # Kernel
    import numpy as np
    from sklearn.neighbors import KernelDensity
    import matplotlib.pyplot as plt

    X = np.random.normal(0, 1, size=(500, 2))

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)

    xmax = 5
    N_points = 250
    x = np.linspace(-xmax, xmax, N_points)
    xx, yy = np.meshgrid(x, x)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    scores = kde.score_samples(xy)
    z = np.exp(scores)
    z = z.reshape(xx.shape)

    z /= np.max(z)

    plt.figure()
    img = plt.imshow(z, cmap='Reds', extent=[-xmax, xmax, -xmax, xmax])
    # plt.scatter(X[:, 0], X[:, 1], s=2, alpha=0.5, color='black')
    plt.colorbar(img)
    plt.xlim([-xmax, xmax])
    plt.ylim([-xmax, xmax])
    plt.xlabel(r'X [mm]')
    plt.ylabel(r'Y [mm]')

    plt.figure()
    plt.plot(x, z[N_points//2, :])
    plt.show()





    fx, fy = stich_fwhm_detector(zosapi=psa, spaxel_scale='60x30', grating='H')

    # max_fwhm = max(np.max(fx), np.max(fy))
    # max_fwhm = np.round(max_fwhm) + 1
    # fig, axes = plt.subplots(2, 2)
    # for i, (fwhm_x, fwhm_y) in enumerate(zip(fx, fy)):
    #     print(i)
    #     ax = axes.flatten()[i]
    #
    #     ax.hist(fwhm_x, bins=np.arange(0, max_fwhm), histtype='step', color='red', label='Along [X]')
    #     ax.hist(fwhm_y, bins=np.arange(0, max_fwhm), histtype='step', color='blue', label='Across [Y]')
    #
    #     # ax.set_title(r'IFU-%s' % ifu_section)
    #     ax.set_xlabel(r'FWHM Geometric PSF [$\mu$m]')
    #     ax.set_ylabel(r'Frequency')
    #     ax.set_xlim([0, np.round(max_fwhm) + 1])
    #     ax.set_xticks(np.arange(0, max_fwhm))
    #
    #     ax.legend()
    # plt.show()


    # analysis = e2e.SpotDiagramDetector(zosapi=psa)
    # spaxel_scale = '60x30'
    # focal_plane = e2e.focal_planes['IFS'][spaxel_scale]['DET']
    # grating = 'H'
    # ifu = 'AB'
    # options = {'which_system': 'IFS', 'AO_modes': [], 'scales': ['60x30'], 'IFUs': [ifu],
    #            'grating': [grating]}
    # wavelength_idx = np.linspace(1, 23, 7).astype(int)
    # list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
    #                                         wavelength_idx=wavelength_idx, configuration_idx=None, surface=focal_plane,
    #                                         N_rays=20)
    #
    # xy, obj_xy, foc_xy, waves = list_results
    #
    # N_waves = wavelength_idx.shape[0]
    # N_configs = 38
    # colors = cm.Reds(np.linspace(0.5, 1, N_waves))
    # x_foc, y_foc = foc_xy[:, :, 0, 0], foc_xy[:, :, 0, 1]
    # a = 100
    # config_list = ['Odd', 'Even']
    #
    # for k in range(2):
    #     fig, ax = plt.subplots(1, 1)
    #     for i in range(N_waves):
    #         for j in range(N_configs):
    #             x, y = xy[i, k + 2*j, 0, :, 0], xy[i, k + 2*j, 0, :, 1]
    #             mx, my = np.mean(x), np.mean(y)
    #             dx = x_foc[i, k + 2*j] + a * (x - mx)
    #             dy = y_foc[i, k + 2*j] + a * (y - my)
    #             ax.scatter(dx, dy, s=2, color=colors[i], marker='+')
    #     ax.set_aspect('equal')
    #     ax.scatter(x_foc[:, k::2], y_foc[:, k::2], s=1, color='black')
    #     ax.set_xlabel(r'X [mm]')
    #     ax.set_ylabel(r'Y [mm]')
    #     ax.grid(True)
    #     xmin = -65 * (-1)**(k) if ifu == 'AB' or ifu =='EF' else -65 * (-1)**(k + 1)
    #     xmax = 0.0
    #     # xmin, xmax = -65 * (-1)**(k + 1), 0 if ifu == ''
    #     ymin, ymax = -35, 35
    #     ax.set_xticks(np.arange(min(xmin, xmax), max(xmin, xmax) + 5, 5))
    #     ax.set_yticks(np.arange(ymin, ymax + 5, 5))
    #     ax.set_xlim([min(xmin, xmax), max(xmin, xmax)])
    #     ax.set_ylim([ymin, ymax])
    #     ax.set_title(r'IFU-%s Config: %s | %s Grating ' % (ifu, config_list[k], grating))
    #
    #     # Add Circle reference
    #     x_cent, y_cent = x_foc[:, k::2][:, -1], y_foc[:, k::2][:, -1]
    #     for i in range(N_waves):
    #         rect = plt.Rectangle(xy=(x_cent[i] - 1.5, y_cent[i] - 1.5), width=3.0, height=3.0, color='b', fill=False)
    #         ax.add_artist(rect)
    #
    #     mng = plt.get_current_fig_manager()
    #     mng.full_screen_toggle()
    #     fig_name = 'DETECTOR_SPOTS_%s_IFU-%s-%s_%s' % (spaxel_scale, ifu, config_list[k], grating)
    #     if os.path.isfile(os.path.join(results_path, fig_name)):
    #         os.remove(os.path.join(results_path, fig_name))
    #     fig.savefig(os.path.join(results_path, fig_name))
    #     plt.close(fig)
    #
    # plt.show()
    #
    #
    # N_waves = wavelength_idx.shape[0]
    # N_configs = 38
    # colors = cm.Reds(np.linspace(0.5, 1, N_waves))
    # n_rows, n_cols = N_waves, N_configs
    # fig, axes = plt.subplots(n_rows, n_cols)
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         ax = axes[i][j]
    #         x, y = xy[i, j, 0, :, 0], xy[i, j, 0, :, 1]
    #         mx, my = np.mean(x), np.mean(y)
    #         dx, dy = 1000 * (x - mx), 1000 * (y - my)  # in microns
    #         ax.scatter(dx, dy, s=3, color=colors[i])
    #         # ax.set_xlabel(r'X [mm]')
    #         # ax.set_ylabel(r'X [mm]')
    #         ax.set_xlim([-25, 25])
    #         ax.set_ylim([-25, 25])
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    #         ax.set_aspect('equal')
    #         ax.xaxis.set_visible('False')
    #         ax.yaxis.set_visible('False')
    plt.show()

    # x, y = XY[j, :, 0], XY[j, :, 1]
    #
    # mx, my = np.mean(x), np.mean(y)
    # dx, dy = 1000*(x - mx), 1000*(y - my)       # in microns
    # ax = axes[j]
    # ax.scatter(dx, dy, s=3, color=color)
    # # ax.set_xlabel(r'X [mm]')
    # # ax.set_ylabel(r'X [mm]')
    # ax.set_xlim([-25, 25])
    # ax.set_ylim([-25, 25])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_aspect('equal')
    # ax.xaxis.set_visible('False')
    # ax.yaxis.set_visible('False')

    # how many rays
    plt.show()

    # # # # # #

    # analysis = e2e.GeometricFWHM_PSF_Analysis(zosapi=psa)
    # spaxel_scale = '60x30'
    # focal_plane = e2e.focal_planes['IFS'][spaxel_scale]['DET']
    # options = {'which_system': 'IFS', 'AO_modes': [], 'scales': ['60x30'], 'IFUs': ['AB'],
    #            'grating': ['HK']}
    # list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
    #                                         wavelength_idx=None, configuration_idx=None, surface=focal_plane,
    #                                         N_rays=30)
    # fwhm, obj_xy, foc_xy, wavelengths = list_results



    if surface == None:     # Detector Plane

        fwhm_x, fwhm_y = fwhm[:, :, :, 1].flatten(), fwhm[:, :, :, 2].flatten()
        max_fwhm = np.max(fwhm[:, :, :, 1:])
        colors = ['red', 'blue']
        direction = ['Along Slice [X]', 'Across Slice [Y]']

        fig, axes = plt.subplots(1, 2)
        for i, data in enumerate([fwhm_x, fwhm_y]):

            ax = axes[i]
            ax.hist(data, histtype='step', color=colors[i])
            ax.set_xlim([0, np.round(max_fwhm) + 1])
            ax.set_xlabel(r'FWHM Geometric PSF [$\mu$m]')
            ax.set_title(direction[i])
            ax.set_ylabel(r'Frequency')

        fig_name =

    #
    # def stitch_fwhm_map(spaxel_scale, N_rays, wavelength_idx, spectral_band='HK', plane='PO', show='focal'):
    #
    #     ifu_sections = ['AB', 'CD', 'EF', 'GH']
    #     # Get the Zemax number for the Surface at which we calculate the RMS WFE
    #     focal_plane = e2e.focal_planes['IFS'][spaxel_scale][plane]
    #     analysis = e2e.GeometricFWHM_PSF_Analysis(zosapi=psa)
    #     fwhm_maps, object_coord, focal_coord = [], [], []
    #     # Loop over the IFU sections
    #     for ifu_section in ifu_sections:
    #         options = {'which_system': 'IFS', 'AO_modes': [], 'scales': [spaxel_scale], 'IFUs': [ifu_section],
    #                    'grating': [spectral_band]}
    #         fwhm, obj_xy, foc_xy, waves = analysis.loop_over_files(files_dir=files_path, files_opt=options,
    #                                                                results_path=results_path, wavelength_idx=wavelength_idx,
    #                                                                configuration_idx=None, surface=focal_plane,
    #                                                                N_rays=N_rays)
    #         fwhm_maps.append(fwhm)
    #         focal_coord.append(foc_xy)
    #         object_coord.append(obj_xy)
    #
    #     fwhm_field = np.concatenate(fwhm_maps, axis=1)
    #     obj_xy = np.concatenate(object_coord, axis=1)
    #     foc_xy = np.concatenate(focal_coord, axis=1)
    #
    #     min_fwhm = np.min(fwhm_field)
    #     max_fwhm = np.max(fwhm_field)
    #
    #     # for j_wave, wavelength in enumerate(waves):
    #     #
    #     #     if show == 'object':
    #     #         x, y = obj_xy[j_wave, :, :, 0].flatten(), obj_xy[j_wave, :, :, 1].flatten()
    #     #         axis_label = 'Object Plane'
    #     #         ref = 'OBJ'
    #     #     elif show == 'focal':
    #     #         x, y = foc_xy[j_wave, :, :, 0].flatten(), foc_xy[j_wave, :, :, 1].flatten()
    #     #         axis_label = 'Focal Plane'
    #     #         ref = 'FOC'
    #     #     else:
    #     #         raise ValueError("'show' must be either 'object' or 'focal'")
    #     #
    #     #     triang = tri.Triangulation(x, y)
    #     #
    #     #     fig1, ax = plt.subplots(1, 1)
    #     #     ax.set_aspect('equal')
    #     #     tpc = ax.tripcolor(triang, fwhm_field[j_wave].flatten(), shading='flat', cmap='jet')
    #     #     tpc.set_clim(vmin=min_fwhm, vmax=max_fwhm)
    #     #     ax.set_xlabel(axis_label + r' X [mm]')
    #     #     ax.set_ylabel(axis_label + r' Y [mm]')
    #     #     title = r'RMS Field Map Stitched IFU | %s mas %s %.3f $\mu$m' % (spaxel_scale, spectral_band, wavelength)
    #     #     ax.set_title(title)
    #     #     plt.colorbar(tpc, ax=ax)
    #     #     fig_name = "GEO_FWHM_PSF_%s_SPEC_%s_%s_SURF_%s_WAVE%d" % (spaxel_scale, spectral_band, ref, plane, wavelength_idx[j_wave])
    #     #     if os.path.isfile(os.path.join(results_path, fig_name)):
    #     #         os.remove(os.path.join(results_path, fig_name))
    #     #     fig1.savefig(os.path.join(results_path, fig_name))
    #
    #     return fwhm_field, foc_xy, fwhm_maps, focal_coord
    #
    # fwhm_field, foc_xy, fwhm_maps, focal_coord = stitch_fwhm_map(spaxel_scale='60x30', N_rays=20, wavelength_idx=[1],
    #                                                              spectral_band='VIS', plane='PO', show='focal')
    #
    # x, y = foc_xy[:, :, :, 0].flatten(), foc_xy[:, :, :, 1].flatten()
    # triang = tri.Triangulation(x, y)
    # fig1, ax = plt.subplots(1, 1)
    # ax.set_aspect('equal')
    # tpc = ax.tripcolor(triang, fwhm[:, :, :, 0].flatten(), shading='flat', cmap='jet')
    # tpc.set_clim(vmin=0, vmax=5*np.median(fwhm))
    # ax.set_xlabel(axis_label + r' X [mm]')
    # ax.set_ylabel(axis_label + r' Y [mm]')
