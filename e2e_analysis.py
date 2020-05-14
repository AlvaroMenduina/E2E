"""

Performance Analysis for the End-to-End Models

author: Alvaro Menduina
date: March 2020

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.tri as tri
import matplotlib.mlab as mlab
import seaborn as sns
from time import time
import h5py
import datetime
from sklearn.neighbors import KernelDensity

from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import constants
from win32com.client import CastTo

# Get the module version from Git
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

# Parameters

# We need to know the Zemax surface number for all Focal Planes
# for each mode (ELT, HARMONI, IFS) and for each spaxel scale (4x4, 10x10, 20x20, 60x30)
# for the FPRS, PO, IFU, SPEC focal planes
focal_planes = {}
focal_planes['IFS'] = {'4x4': {'AB': {'FPRS': 6, 'PO': 41, 'IS': 70, 'SL': 88, 'DET': None},
                               'CD': {'FPRS': 6, 'PO': 41, 'IS': 69, 'SL': 88, 'DET': None},
                               'EF': {'FPRS': 6, 'PO': 41, 'IS': 70, 'SL': 88, 'DET': None},
                               'GH': {'FPRS': 6, 'PO': 41, 'IS': 69, 'SL': 88, 'DET': None}},
                       '10x10': {'AB': {'FPRS': 6, 'PO': 38, 'IS': 67, 'SL': 85, 'DET': None},
                                 'CD': {'FPRS': 6, 'PO': 38, 'IS': 66, 'SL': 85, 'DET': None},
                                 'EF': {'FPRS': 6, 'PO': 38, 'IS': 67, 'SL': 85, 'DET': None},
                                 'GH': {'FPRS': 6, 'PO': 38, 'IS': 66, 'SL': 85, 'DET': None}},
                       '20x20': {'AB': {'FPRS': 6, 'PO': 38, 'IS': 67, 'SL': 85, 'DET': None},
                                 'CD': {'FPRS': 6, 'PO': 38, 'IS': 66, 'SL': 85, 'DET': None},
                                 'EF': {'FPRS': 6, 'PO': 38, 'IS': 67, 'SL': 85, 'DET': None},
                                 'GH': {'FPRS': 6, 'PO': 38, 'IS': 66, 'SL': 85, 'DET': None}},
                       '60x30': {'AB': {'FPRS': 6, 'PO': 30, 'IS': 59, 'SL': 77, 'DET': None},
                                 'CD': {'FPRS': 6, 'PO': 30, 'IS': 58, 'SL': 77, 'DET': None},
                                 'EF': {'FPRS': 6, 'PO': 30, 'IS': 59, 'SL': 77, 'DET': None},
                                 'GH': {'FPRS': 6, 'PO': 30, 'IS': 58, 'SL': 77, 'DET': None}}}
# Keywords for Focal Plane are:
# PO: PreOptics, IS: Image Slicer, SL: Slit, DET: Detector

# Taken from Zemax example code
class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self):
        # make sure the Python wrappers are available for the COM client and
        # interfaces
        EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)

        # EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        # Note - the above can also be accomplished using 'makepy.py' in the
        # following directory:
        #      {PythonEnv}\Lib\site-packages\wind32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #      {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*

        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException(
                "Unable to intialize COM connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException(
                "License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if (self.TheApplication.LicenseStatus is
                constants.LicenseStatusType_PremiumEdition):
            return "Premium"
        elif (self.TheApplication.LicenseStatus is
              constants.LicenseStatusType_ProfessionalEdition):
            return "Professional"
        elif (self.TheApplication.LicenseStatus is
              constants.LicenseStatusType_StandardEdition):
            return "Standard"
        else:
            return "Invalid"


def read_hdf5(path_to_file):
    """
    Read a HDF5 file containing some analysis results

    HDF5 files for the E2E analysis have the following 'Group' structure:

        (1) "Zemax Metadata": this Group is a dictionary containing metadata on the Zemax file used in the analysis
            this includes: file name, system mode, spaxel scale, IFU channel, grating, AO modes
            Plus the Git hash for the E2E Python code [for traceability]

        (2) Analysis Name 1: each analysis type (RMS_WFE, FWHM_PSF) will have an associated group. Within each group,
            there will be an arbitrary number of subgroups.
                - Subgroup #0: each subgroup represents an instance of that analysis type; for example an RMS WFE
                  ran at a specific surface, with a set of parameters. Each subgroup will containing a series of
                  datasets representing the results of the analysis, plus some associated metadata (surface, parameters)
                - Subgroup #N: another analysis ran at a different surface, or with different sampling
        (3) Analysis Name 2: another type of analysis
                - [...]

    TODO: at the moment this function only reads the metadata for Zemax and each Analysis.

    :param path_to_file:
    :return:
    """

    print("\nReading HDF5 file: ", path_to_file)
    file = h5py.File(path_to_file, 'r')

    # List the groups
    groups = list(file.keys())
    print("Groups available: ", groups)

    # Read Zemax Metadata
    zemax_metadata = {}
    print("\nZemax Metadata:")
    for key in file['Zemax Metadata'].attrs.keys():
        print('{} : {}'.format(key, file['Zemax Metadata'].attrs[key]))
        zemax_metadata[key] = file['Zemax Metadata'].attrs[key]

    # Read the analysis groups
    for group_name in groups:
        if group_name != 'Zemax Metadata':
            analysis_group = file[group_name]
            print('\nAnalysis: ', group_name)
            # For each Analysis Group we loop over subgroups
            for subgroup_key in analysis_group.keys():
                subgroup = analysis_group[subgroup_key]
                print('Subgroup #', subgroup_key)
                # List the metadata of the subgroup
                for att_key in subgroup.attrs.keys():
                    print('     {} : {}'.format(att_key, subgroup.attrs[att_key]))

    file.close()

    return zemax_metadata


def get_fields(system, info=False):
    """
    Read the Field Points for a given Optical System
    :param system: a system loaded with the Python Standalone Application
    :param info: boolen, whether to print the information
    :return: field_points, an array of size [N_fields, 2]
    """

    system_data = system.SystemData
    fields = system_data.Fields
    N_fields = fields.NumberOfFields

    if info is True:
        print("\nReading Field Points")
        print("Total Number of Field Points: %d" % N_fields)

    field_points = np.zeros((N_fields, 2))

    for k in np.arange(1, N_fields + 1):
        _field = fields.GetField(k)
        field_points[k - 1, 0] = _field.X
        field_points[k - 1, 1] = _field.Y

        if info is True:
            print("X = %.4f, Y = %.4f" % (_field.X, _field.Y))

    return field_points


def get_wavelengths(system, info=False):
    """
    Read the Wavelengths for a given Optical System
    :param system: a system loaded with the Python Standalone Application
    :param info: boolen, whether to print the information
    :return: field_points, an array of size [N_wavelengths]
    """

    system_data = system.SystemData
    wavelengths = system_data.Wavelengths
    N_wavelengths = wavelengths.NumberOfWavelengths

    if info is True:
        print("\nReading Wavelengths")
        print("Total Number of Wavelengths: %d" % N_wavelengths)

    wavelength_array = np.zeros(N_wavelengths)

    for k in np.arange(1, N_wavelengths + 1):
        _wave = wavelengths.GetWavelength(k).Wavelength
        wavelength_array[k - 1] = _wave

        if info is True:
            print("%.5f microns" % _wave)

    return wavelength_array


def create_zemax_file_list(which_system,
                           AO_modes=('NOAO', 'LTAO', 'SCAO', 'HCAO'),
                           scales=('4x4', '10x10', '20x20', '60x30'),
                           IFUs=('AB',),
                           grating=('VIS', 'IZJ', 'HK', 'IZ', 'J', 'H', 'K', 'Z_HIGH', 'H_HIGH', 'K_LONG', 'K_SHORT')):
    """
    Creates a list of Zemax file names using the naming convention that Matthias defined for the E2E model

    ['CRYO_PO4x4_IFU-AB_SPEC-VIS.zmx',
    'CRYO_PO4x4_IFU-AB_SPEC-IZJ.zmx',
    'CRYO_PO4x4_IFU-AB_SPEC-HK.zmx',
    ...,]

    Output: file_list, and settings_list

    settings_list is a list of dictionaries containing the setting (scale, IFU, spectral band) of each file
    we need that in the future to postprocess the results of each Zemax file

    :param which_system: which system do we want to analyze, can be 'ELT' or 'HARMONI' or 'IFS'
    :param AO_modes: a list of AO modes that can be ['NOAO', 'LTAO', 'SCAO', 'HCAO']
    :param scales: a list of Spaxel Scales that can be ['4x4', '10x10', '20x20', '60x30']
    :param IFUs: a list of IFU subfields that can be ['AB', ...]
    :param grating: a list of Gratings that can be ['VIS', 'IZJ', 'HK', 'IZ', 'J', 'H', 'K', 'Z_HIGH', 'H_HIGH', 'K_LONG', 'K_SHORT']
    :return:
    """

    if which_system not in ['ELT', 'HARMONI', 'IFS']:
        raise ValueError("System must be 'ELT' or 'HARMONI' or 'IFS'")

    if which_system == 'IFS':     # Only the IFS, starting at the CRYO

        file_list = []
        settings_list = []          # List of settings dictionaries
        for scale in scales:
            name_scale = 'CRYO_PO' + scale
            for ifu in IFUs:
                if scale != '4x4':
                    name_IFU = '_IFU' + ifu
                else:
                    name_IFU = '_IFU-' + ifu            # For 4x4 the IFU is reversed
                for spec in grating:

                    if ifu == 'AB' or ifu == 'EF':
                        # For 4x4 the SPEC is also reversed, we add a '-' sign
                        name_spec = '_SPEC-' + spec if scale == '4x4' else '_SPEC' + spec

                    if ifu == 'CD' or ifu == 'GH': # _SPEC-K_LONG
                        name_spec = '_SPEC' + spec if scale == '4x4' else '_SPEC-' + spec

                    filename = name_scale + name_IFU + name_spec + '.zmx'
                    file_list.append(filename)

                    # Save settings:
                    sett_dict = {'system': which_system, 'scale': scale, 'ifu': ifu, 'grating': spec}
                    settings_list.append(sett_dict)

        return file_list, settings_list

    if which_system == 'HARMONI':     # The same as 'IFS' but including the FPRS with the AO_modes

        file_list = []
        settings_list = []
        # Create a list of file names from the CRYO onwards
        ifs_file_list, ifs_sett_list = create_zemax_file_list(which_system='IFS', scales=scales, IFUs=IFUs, grating=grating)
        # Add FPRS_ + AO mode as prefix
        for AO_mode in AO_modes:
            name_fprs = 'FPRS_' + AO_mode + '_'
            for ifs_file, ifs_sett in zip(ifs_file_list, ifs_sett_list):
                filename = name_fprs + ifs_file
                file_list.append(filename)

                ifs_sett['AO_mode'] = AO_mode       # Add AO mode to the settings
                ifs_sett['system'] = which_system   # Override the system mode to HARMONI, not IFS
                settings_list.append(ifs_sett)

        return file_list, settings_list

    if which_system == 'ELT':         # The same as 'HARMONI' but including the ELT

        # Create a list of file names from HARMONI
        harmoni_files_list, harmoni_sett_list = create_zemax_file_list(which_system='HARMONI', AO_modes=AO_modes,
                                                                       scales=scales, IFUs=IFUs, grating=grating)
        # Add ELT as prefix
        file_list = ['ELT_' + harmoni_file for harmoni_file in harmoni_files_list]
        settings_list = []
        for harm_sett in harmoni_sett_list:
            harm_sett['system'] = which_system      # Override the system mode to ELT, not HARMONI
            settings_list.append(harm_sett)

        return file_list, settings_list

    return


class AnalysisGeneric(object):

    """
    Object to perform fairly generic analyses on Zemax files created by the E2E model

    If you want to run your own custom analysis the best way is to use Class Inheritance
    Create a child class from this AnalysisGeneric object
    Define a method (function, typically static method) in the child class representing the 'analysis_function'
    that self.run_analysis will call. That method / function should perform all the operations that
    you are interested in for your analysis

    """

    def __init__(self, zosapi, *args, **kwargs):
        """
        We basically need a ZOS API Standalone Application to
        open and close the files we want to analyze
        :param zosapi:
        :param args:
        :param kwargs:
        """

        self.zosapi = zosapi

    def run_analysis(self, analysis_function, results_shapes, results_names, files_dir, zemax_file, results_path,
                     wavelength_idx=None, configuration_idx=None, surface=None,
                     *args, **kwargs):
        """

        Run some analysis on the Optical System over a range of Wavelengths and Configurations
        For this, we need some Python function "analysis_function" that receives as input
        (Optical System, Wavelength Number, Configuration Number, Surface Number, *args, **kwargs)
        *args and **kwargs represent whatever other information our function might need for the
        calculations. This could be something like Number of Fields, Number of Rays, or some other
        Zemax analysis parameters

        The "run_analysis" function will loop over our chosen Wavelengths and Configurations (Slices)
        and apply the specified "analysis_function"

        ~~ Extremely IMPORTANT! ~~
        We are aware that the shape and type of results will vary for each analysis
        In order to accommodate this variability we use the following strategy:

            You can customize the output of 'run_analysis' by using a list 'results_names'
            that contains the names of the variables that your 'analysis_function' will provide
            as well as a list 'results_shapes' containing the shape of such variables

            This code 'run_analysis' will dynamically create such variables using the globals() dictionary
            such that any array of arbitrary shape [N_waves, N_configs, Nx, Ny, ..., Nz] can be
            accommodated.

            For example, if we want to run a Spot Diagram analysis we would call
            'run_analysis' with analaysis_function=SpotDiagram.analysis_function and provide as arguments
            results_names = ['xy'], results_shapes = [(N_fields, N_rays^2, 2)]

            'run_analysis' would then loop over those lists (in this case we only have 1 array)
            and create the global variable xy, with shape [N_waves, N_config, N_fields, N_rays^2, 2]
            and loop over the Wavelengths and Configurations applying the 'analysis_function'

        All of this is done for a given "zemax_file" in the "files_dir". This is a template so if you
        want to run the analysis over multiple files you can create a Child Class and add a method
        that loops over the files (see RMS_ XXX)

        :param analysis_function: arbitrary function that does the calculations needed for our analyses
        :param results_shapes: list of shapes for the arrays that 'analysis_function' produces
        :param results_shape: list of names of the arrays that 'analysis_function' produces
        :param files_dir: path to where you keep the Zemax files
        :param zemax_file: the name of the Zemax file you want to analyze
        :param results_path: path where you will save the results
        :param wavelength_idx: list of Zemax wavelength numbers. If left as None, it will analyze all Wavelengths
        :param configuration_idx: list of Zemax configurations. If left as None, it will analyze all Configurations
        :param surface: zemax surface number at which we will compute the analysis. If left as None, it will use the Image Plane
        :param args: whatever other arguments your "analysis_function" may need
        :param kwargs: whatever other keyword arguments your "analysis_function" may need
        :return:
        """

        # check that the file name is correct and the zemax file exists
        if os.path.exists(os.path.join(files_dir, zemax_file)) is False:
            raise FileExistsError("%s does NOT exist" % zemax_file)

        print("\nOpening Zemax File: ", zemax_file)
        self.zosapi.OpenFile(os.path.join(files_dir, zemax_file), False)
        file_name = zemax_file.split(".")[0]            # Remove the ".zmx" suffix

        # Check if the results directory already exists
        results_dir = os.path.join(results_path, file_name)
        print("Results will be saved in: ", results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)           # If not, create the directory to store results

        # Get some info on the system
        system = self.zosapi.TheSystem                      # The Optical System
        MCE = system.MCE                                    # Multi Configuration Editor
        LDE = system.LDE                                    # Lens Data Editor
        N_surfaces = LDE.NumberOfSurfaces
        N_configs = MCE.NumberOfConfigurations

        print("\nSystem Data:")
        print("Number of Surfaces: ", N_surfaces)
        print("Number of Configurations / Slices: ", N_configs)
        _wavelengths = get_wavelengths(system, info=False)
        N_waves = _wavelengths.shape[0]
        # field_points = get_fields(system, info=False)

        # Let's deal with the data format
        if wavelength_idx is None:         # We use all the available wavelengths
            wavelengths = _wavelengths
            wavelength_idx = np.arange(1, N_waves + 1)
        else:     # We have received a list of wavelength indices
            wavelengths = [_wavelengths[idx - 1] for idx in wavelength_idx]
            N_waves = len(wavelength_idx)

        if configuration_idx is None:
            configurations = np.arange(1, N_configs + 1)
        else:       # We have received a list of configurations
            configurations = configuration_idx
            N_configs = len(configuration_idx)
        N_slices = N_configs

        if surface is None:     # we go to the final image plane
            surface = system.LDE.NumberOfSurfaces - 1

        N_surfaces = system.LDE.NumberOfSurfaces
        if surface != (N_surfaces - 1):
            # The Surface is not the final Image Plane. We must force Zemax to IGNORE
            # all surfaces after our surface of interest so that the RWCE operands works
            surfaces_to_ignore = np.arange(surface + 1, N_surfaces)
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = True

            last_surface = system.LDE.GetSurfaceAt(surface)
            thickness = last_surface.Thickness
            print("Thickness: ", last_surface.Thickness)
            # image = system.LDE.GetSurfaceAt(N_surfaces - 1)
            # print("\nImage Name: ", image.Comment)

            # Temporarily set the Thickness to 0
            last_surface.Thickness = 0.0
            print("Thickness: ", last_surface.Thickness)

        # Double check we are using the right surface
        print("\nSurface Name: ", system.LDE.GetSurfaceAt(surface).Comment)

        # Customize the initilization of the results array so that we can vary the shapes
        # some analysis may be [spaxels_per_slice, N_slices, N_waves] such as an RMS WFE map
        # others might be [N_fields, N_rays, N_slices, N_waves] such as a Spot Diagram

        print("\nDynamically creating Global Variables to store results")
        for _name, _shape in zip(results_names, results_shapes):
            print("Variable: %s with shape (N_waves, N_configs) + " % _name, _shape)
            globals()[_name] = np.empty((N_waves, N_slices) + _shape)
            # print(globals()[_name].shape)

        print("\nApplying 'analysis_function': ", analysis_function.__name__)
        print("At Surface #%d | Image Plane is #%d" % (surface, N_surfaces - 1))
        print("For Wavelength Numbers: ", wavelength_idx)
        print("For Configurations %d -> %d" % (configurations[0], configurations[-1]))

        start = time()
        print("\nAnalysis Started")
        for k, (wave_idx, wavelength) in enumerate(zip(wavelength_idx, wavelengths)):
            print("Wavelength #%d: %.3f" % (wave_idx, wavelength))

            for j, config in enumerate(configurations):

                # print("Wavelength #%d: %.3f | config: %d" %(wave_idx, wavelength, config))

                # Here is where the arbitrary "analysis_function" will be called for each Wavelength and Configuration
                results = analysis_function(system=system, wave_idx=wave_idx, config=config, surface=surface,
                                            *args, **kwargs)

                # Store the result values in the arrays:
                for _name, res in zip(results_names, results):
                    globals()[_name][k, j] = res
                # data_results[:, j, k] = data            # The results of the analysis
                # object_xy[:, :, j, k] = objxy           # The field positions for the OBJECT
                # focal_xy[:, :, j, k] = focxy            # The raytrace results at the chosen Focal Plane

        # Unignore the surfaces
        if surface != (N_surfaces - 1):
            last_surface = system.LDE.GetSurfaceAt(surface)
            last_surface.Thickness = thickness
            for surf_number in surfaces_to_ignore:
                _surf = system.LDE.GetSurfaceAt(surf_number)
                _surf.TypeData.IgnoreSurface = False

        self.zosapi.CloseFile(save=False)
        analysis_time = time() - start
        print("\nAnalysis done in %.1f seconds" % analysis_time)
        time_per_wave = analysis_time / len(wavelengths)
        print("%.2f seconds per Wavelength" % time_per_wave)

        # local_keys = [key for key in globals().keys()]
        # print(local_keys)

        # We return a list containing the results, and the value of the wavelengths
        list_results = [globals()[_name] for _name in results_names]
        list_results.append(wavelengths)

        return list_results

    def template_analysis_function(self, system, wave_idx, config, surface, *args, **kwargs):
        """
        Template for the analysis function

        Typically we pass as arguments the 'system' so that any information about it can be accessed
        The Zemax Wavelength number 'wave_idx' and Configuration 'config' at which we will apply the analysis
        And the Zemax surface number 'surface' at which we will apply the analysis

        We also include *args and **kwargs to accommodate other parameters your analysis may need
        For instance, the number of rings to be used in RMS wavefront calculations

        :param system:
        :param wave_idx:
        :param config:
        :param surface:
        :param args:
        :param kwargs:
        :return:
        """

        # Write your function operations here

        # This will typically contain a lot of ZOS API commands such as:
        # - Setting the Current configuration:          system.MCE.SetCurrentConfiguration(config)
        # - Reading the Field Points:                   sysField = system.SystemData.Fields
        #                                               fx = sysField.GetField(1).X
        #                                               fy = sysField.GetField(1).Y
        # - Running some Ray Trace:                     raytrace = system.Tools.OpenBatchRayTrace()
        #                                               normUnPolData = raytrace.CreateNormUnpol(N_rays,
        #                                                                        constants.RaysType_Real, toSurface)


        # You should return a list of arrays

        return

    def save_hdf5(self, analysis_name, analysis_metadata, list_results, results_names, file_name, file_settings, results_dir):

        # First thing is to create a separate folder within the results directory for this analysis
        hdf5_dir = os.path.join(results_dir, 'HDF5')
        print("Analysis Results will be saved in folder: ", hdf5_dir)
        if not os.path.exists(hdf5_dir):
            os.mkdir(hdf5_dir)           # If not, create the directory to store results

        hdf5_file = file_name + '.hdf5'
        # Check whether the file already exists
        if os.path.isfile(os.path.join(hdf5_dir, hdf5_file)):   # Overwrite it
            print("HDF5 file already exists. Adding analysis")
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'r+') as f:

                # Here we have 2 options: either we are adding another analysis type or we are adding the same type
                # but with different settings (such as at a different surface)
                file_keys = list(f.keys())
                if analysis_name in file_keys:
                    print("Analysis type already exists")
                    analysis_group = f[analysis_name]
                    # we need to know how many analyses of the same type already exist
                    subgroup_keys = list(analysis_group.keys())
                    subgroup_number = len(subgroup_keys)        # if [0, 1] already exist we call it '2'
                    subgroup = analysis_group.create_group(str(subgroup_number))
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

                else:       # It's a new analysis type

                    # (2) Create a Group for this analysis
                    analysis_group = f.create_group(analysis_name)
                    # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                    subgroup = analysis_group.create_group('0')
                    # Save results datasets
                    for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                        data = subgroup.create_dataset(array_name, data=array)
                    # Save analysis metadata
                    subgroup.attrs['E2E Python Git Hash'] = sha
                    subgroup.attrs['Analysis Type'] = analysis_name
                    date_created = datetime.datetime.now().strftime("%c")
                    subgroup.attrs['Date'] = date_created
                    subgroup.attrs['At Surface #'] = file_settings['surface']
                    # Add whatever extra metadata we have:
                    for key, value in analysis_metadata.items():
                        subgroup.attrs[key] = value

        else:       # File does not exist, we create it now
            print("Creating HDF5 file: ", hdf5_file)
            with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'w') as f:

                # (1) Save Zemax Metadata
                zemax_metadata = f.create_group('Zemax Metadata')
                zemax_metadata.attrs['(1) Zemax File'] = file_name
                zemax_metadata.attrs['(2) System Mode'] = file_settings['system']
                zemax_metadata.attrs['(3) Spaxel Scale'] = file_settings['scale']
                zemax_metadata.attrs['(4) IFU'] = file_settings['ifu']
                zemax_metadata.attrs['(5) Grating'] = file_settings['grating']
                AO = file_settings['AO_mode'] if 'AO_mode' in list(file_settings.keys()) else 'NA'
                zemax_metadata.attrs['(6) AO Mode'] = AO

                # (2) Create a Group for this analysis
                analysis_group = f.create_group(analysis_name)

                # (3) Create a Sub-Group so that we can have the same analysis at multiple surfaces / wavelength ranges
                subgroup = analysis_group.create_group('0')
                # Save results datasets
                for array, array_name in zip(list_results, results_names + ['WAVELENGTHS']):
                    data = subgroup.create_dataset(array_name, data=array)
                # Save analysis metadata
                subgroup.attrs['E2E Python Git Hash'] = sha
                subgroup.attrs['Analysis Type'] = analysis_name
                date_created = datetime.datetime.now().strftime("%c")
                subgroup.attrs['Date'] = date_created
                subgroup.attrs['At Surface #'] = file_settings['surface']
                # Add whatever extra metadata we have:
                for key, value in analysis_metadata.items():
                    subgroup.attrs[key] = value

        return

    def plot_and_save(self, analysis_name, list_results, file_name, file_settings, results_dir, wavelength_idx):
        """
        Semi-"generic" plot and save routine to post-process the analysis results
        It works for analysis types such as RMS Wavefront Error where you have a list of arrays like
        [MAIN_RESULTS, OBJ_XY, FOC_XY, WAVELENGTHS]
        and MAIN_RESULTS is something like the RMS WFE, of shape [N_waves, N_configs, N_fields]
        i.e. a given value for each Field Point, at each Configuration (slice) and Wavelength

        :param analysis_name: string, to add the analysis name to the plots
        :param list_results: list containing the arrays [MAIN_RESULTS, OBJ_XY, FOC_XY, WAVELENGTHS]
        :param file_name: string, name of the Zemax file
        :param file_settings: dictionary of settings, so we can access things like the spaxel scale, the IFU section...
        :param results_dir: path to where the results will be saved
        :param wavelength_idx: list of wavelength numbers used, or None for All Wavelengths
        :return:
        """

        # First thing is to create a separate folder within the results directory for this analysis
        analysis_dir = os.path.join(results_dir, analysis_name)
        print("Analysis Results will be saved in folder: ", analysis_dir)
        if not os.path.exists(analysis_dir):
            os.mkdir(analysis_dir)           # If not, create the directory to store results

        analysis_array = list_results[0]  # The first is always the Analysis results
        object_coord = list_results[1]
        focal_coord = list_results[2]
        wavelengths = list_results[-1]

        N_waves = analysis_array.shape[0]

        if wavelength_idx is None:
            wave_idx = np.arange(1, N_waves + 1)
        else:
            wave_idx = wavelength_idx

        # Read the settings
        system = file_settings['system']
        surface_number = file_settings['surface']
        scale, ifu, grating = file_settings['scale'], file_settings['ifu'], file_settings['grating']
        ao_mode = file_settings['AO_mode'] if 'AO_mode' in list(file_settings.keys()) else None
        # Find out the NameCode for the surface
        surface_codes = focal_planes[system][scale]
        surface_name = list(surface_codes.keys())[list(surface_codes.values()).index(surface_number)]
        # surface_number = str(surface) if surface is not None else '_IMG'

        print("Metadata Review")
        print("Filename: ", file_name)
        print("System: ", system)
        print("Scale: ", scale)
        print("IFU: ", ifu)
        print("Grating: ", grating)
        print("AO mode: ", ao_mode)
        print("Surface Code: ", surface_name)
        print("Surface Number: ", surface_number)

        ### ---------------------------------- Write Results File -------------------------------------------------- ###
        name_code = file_name + '_' + analysis_name + '_SURF_' + surface_name

        # Save Metadata of the Analysis
        output_name = os.path.join(analysis_dir, name_code + "_METADATA.txt")
        metadata_file = open(output_name, "w")
        metadata_file.write("Analysis: %s \n" % analysis_name)
        metadata_file.write("Zemax File: %s \n" % file_name)
        metadata_file.write("At Surface: #%s, %s \n" % (str(surface_number), surface_name))
        metadata_file.write("System Mode: %s \n" % system)
        metadata_file.write("Spaxel Scale: %s \n" % scale)
        metadata_file.write("IFU section: %s \n" % ifu)
        metadata_file.write("AO mode: %s \n" % ao_mode)
        metadata_file.write("Wavelength Numbers: [")
        for idx in wave_idx:
            metadata_file.write(" %d " % idx)
        metadata_file.write(" ] \n")
        metadata_file.write("Wavelengths [microns] : [")
        for wave in wavelengths:
            metadata_file.write(" %.4f " % wave)
        metadata_file.write(" ] \n")

        # Save the arrays
        np.save(os.path.join(analysis_dir, name_code), analysis_array)
        np.save(os.path.join(analysis_dir, name_code + '_OBJECT_COORD'), object_coord)
        np.save(os.path.join(analysis_dir, name_code + '_FOCAL_COORD'), focal_coord)
        np.save(os.path.join(analysis_dir, name_code + '_WAVELENGTHS'), wavelengths)

        # L = ["This is Delhi \n", "This is Paris \n", "This is London \n"]
        #
        # # \n is placed to indicate EOL (End of Line)
        # metadata_file.write("Hello \n")
        # metadata_file.writelines(L)
        metadata_file.close()


        ### --------------------------------- PLOTS and FIGURES --------------------------------------------------- ###

        # Statistics
        means, stds, mins, maxs = np.zeros(N_waves), np.zeros(N_waves), np.zeros(N_waves), np.zeros(N_waves)
        for i in range(N_waves):

            mu0 = np.mean(analysis_array[i])
            std0 = np.std(analysis_array[i])
            means[i], stds[i] = mu0, std0
            mins[i] = np.min(analysis_array[i])
            maxs[i] = np.max(analysis_array[i])

            if i in [0, N_waves//2, N_waves - 1]:    # Only show Min | Central | Max wavelengths

                # fig = plt.figure()
                # plt.hist(analysis_array[i].flatten(), bins=10, histtype='step')
                # plt.xlabel(analysis_name)
                # plt.ylabel(r'Frequency')
                # plt.xlim([mu0 - 5 * std0, mu0 + 5 * std0])
                # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_WAVE%d' % (wave_idx[i])
                # plt.title(fig_name)
                # if os.path.isfile(os.path.join(analysis_dir, fig_name)):
                #     os.remove(os.path.join(analysis_dir, fig_name))
                # fig.savefig(os.path.join(analysis_dir, fig_name))
                # plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                sns.distplot(analysis_array[i].flatten(), ax=ax, axlabel=analysis_name)
                fig_name = file_name + '_' + analysis_name + '_SURF_' + surface_name + '_WAVE%d' % (wave_idx[i])
                # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_WAVE%d' % (wave_idx[i])
                plt.title(fig_name)
                if os.path.isfile(os.path.join(analysis_dir, fig_name)):
                    os.remove(os.path.join(analysis_dir, fig_name))
                fig.savefig(os.path.join(analysis_dir, fig_name))
                plt.close(fig)

        fig = plt.figure()
        plt.plot(wavelengths, means, color='black', label='Mean')
        plt.fill_between(wavelengths, means - stds, means + stds, alpha=0.2, color='salmon', label=r'$\pm \sigma$')
        plt.plot(wavelengths, maxs, color='crimson', label='Max')
        plt.plot(wavelengths, mins, color='blue', label='Min')
        plt.legend()
        plt.ylim(bottom=0)
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(analysis_name)
        fig_name = file_name + '_' + analysis_name + '_SURF' + surface_name
        plt.title(fig_name)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        fig.savefig(os.path.join(analysis_dir, fig_name))
        plt.close(fig)

        # (1) Object Field Coordinates [wavelengths, configs, N_fields, xy]
        # Independent of wavelength
        plt.figure()
        plt.scatter(object_coord[0, :, :, 0], object_coord[0, :, :, 1], s=3)
        plt.xlabel(r'Object X [mm]')
        plt.ylabel(r'Object Y [mm]')
        plt.axes().set_aspect('equal')
        fig_name = file_name + '_XY_OBJECT'
        plt.title(fig_name)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        plt.savefig(os.path.join(analysis_dir, fig_name))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

        # (2) Focal Plane
        # Positions might depend on Wavelength (at the detector)
        # so we show all wavelengths
        colors = cm.Reds(np.linspace(0.5, 1, N_waves))
        plt.figure()
        for i in range(N_waves):
            plt.scatter(focal_coord[i, :, :, 0], focal_coord[i, :, :, 1], s=3, color=colors[i])
        plt.xlabel(r'Focal X Coordinate [mm]')
        plt.ylabel(r'Focal Y Coordinate [mm]')
        fig_name = file_name + '_XY_FOCAL_SURF' + surface_name
        plt.title(fig_name)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        plt.savefig(os.path.join(analysis_dir, fig_name))
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

        # (3) Analysis Results
        # This might be very tricky
        # Reference: both Object and Focal coordinates

        # (A) Object Coordinates
        # Independent of wavelength
        i_wave = 0
        x, y = object_coord[i_wave, :, :, 0].flatten(), object_coord[i_wave, :, :, 1].flatten()
        triang = tri.Triangulation(x, y)
        fig, ax = plt.subplots(1, 1)
        tpc = ax.tripcolor(triang, analysis_array[i_wave].flatten(), shading='flat', cmap='jet')
        ax.scatter(x, y, s=2, color='black')
        ax.set_xlabel(r'Object X [mm]')
        ax.set_ylabel(r'Object Y [mm]')
        ax.set_aspect('equal')
        plt.colorbar(tpc, ax=ax, orientation='horizontal')
        fig_name = file_name + '_' + analysis_name + '_SURF_' + surface_name + '_OBJ_WAVE%d' % (wave_idx[i_wave])
        # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_OBJ_WAVE%d' % (wave_idx[i_wave])
        ax.set_title(fig_name)
        if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            os.remove(os.path.join(analysis_dir, fig_name))
        plt.savefig(os.path.join(analysis_dir, fig_name))
        # plt.show(block=False)
        # plt.pause(0.1)
        plt.close()

        # (B) Focal Coordinates

        if surface_number != None:     # Not a Detector Plane -> Focal has the same position for all wavelengths?

            # We have to loop over wavelengths
            for i in [0, N_waves // 2, N_waves - 1]:
                x, y = focal_coord[i, :, :, 0].flatten(), focal_coord[i, :, :, 1].flatten()
                triang = tri.Triangulation(x, y)
                fig, ax = plt.subplots(1, 1)
                tpc = ax.tripcolor(triang, analysis_array[i].flatten(), shading='flat', cmap='jet')
                ax.scatter(x, y, s=2, color='black')
                ax.set_xlabel(r'Focal X [mm]')
                ax.set_ylabel(r'Focal Y [mm]')
                ax.set_aspect('equal')
                plt.colorbar(tpc, ax=ax, orientation='horizontal')
                fig_name = file_name + '_' + analysis_name + '_SURF_' + surface_name + '_FOC_WAVE%d' % (wave_idx[i])
                # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_FOC_WAVE%d' % (wave_idx[i])
                ax.set_title(fig_name)
                if os.path.isfile(os.path.join(analysis_dir, fig_name)):
                    os.remove(os.path.join(analysis_dir, fig_name))
                plt.savefig(os.path.join(analysis_dir, fig_name))
                # plt.show(block=False)
                # plt.pause(0.1)
                plt.close()

        elif surface_number == None:  # Detector Plane -> Focal will depend on wavelength

            # Average across spaxel scales?
            x, y = focal_coord[:, :, :, 0], focal_coord[:, :, :, 1]
            x, y = np.mean(x, axis=2).flatten(), np.mean(y, axis=2).flatten()
            analysis_array = np.mean(analysis_array, axis=2)


            # (B) Focal Coordinates
            # All Wavelengths at once
            # x, y = focal_coord[:, :, :, 0].flatten(), focal_coord[:, :, :, 1].flatten()
            triang = tri.Triangulation(x, y)
            fig, ax = plt.subplots(1, 1)
            tpc = ax.tripcolor(triang, analysis_array.flatten(), shading='flat', cmap='jet')
            # ax.scatter(x, y, s=2, color='black')
            ax.set_xlabel(r'Focal Plane X [mm]')
            ax.set_ylabel(r'Focal Plane Y [mm]')
            ax.set_aspect('equal')
            plt.colorbar(tpc, ax=ax, orientation='horizontal')
            fig_name = file_name + '_' + analysis_name + '_SURF_' + surface_name + '_FOC'
            # fig_name = file_name + '_SURF' + surface_number + analysis_name + '_FOC'
            ax.set_title(fig_name)
            if os.path.isfile(os.path.join(analysis_dir, fig_name)):
                os.remove(os.path.join(analysis_dir, fig_name))
            plt.savefig(os.path.join(analysis_dir, fig_name))
            # plt.show(block=False)
            # plt.pause(0.1)
            plt.close()

        return


class SpotDiagramAnalysis(AnalysisGeneric):
    """
    Example of a custom analysis - Spot Diagrams

    This is how it works:
        - We inherit the AnalysisGeneric class
        - We define analysis_function_spot_diagram as the 'analysis_function'
        - self.run_analysis will loop over Wavelengths and Configurations calling analysis_function_spot_diagram
        - self.loop_over_files will loop over the E2E model files, calling run_analysis and saving plots

    """

    @staticmethod
    def analysis_function_spot_diagram(system, wave_idx, config, field, surface, N_rays):
        """
        Calculate the Spot Diagram for a given Wavelength number and Configuration
        for a list of Field Points (field), at a specific Surface

        For the Pupil Sampling we do the following
        we sample with a rectangular grid for P_x between [-1, 1] with N_rays
        which gives an XY grid for (P_x, P_y) with a total of N_rays^2
        Obviously, some rays lie outside the pupil (P_x ^ 2 + P_y ^ 2) > 1
        so those rays are ignored in the Ray Tracing. We only trace meaningful rays!

        But in order to avoid the further complication of finding out how many rays
        actually make sense, which will change as a function of the N_rays we choose
        and would force us to adapt the size of the arrays a posteriori
        we always keep the results as [N_rays ^ 2] in size, and fill with NaN
        those rays which will not be traced.
        This is not a problem because Python deals with NaN values very well.
        We have np.nanmax(), np.nanmean() functions, and the Spot Diagram plots
        will automatically ignore any NaN value.

        This function will return a list of arrays, containing a single entry
        an array of shape [N_fields, N_rays^2, 2] containing the (X, Y) coordinates
        of each ray for each field

        :param system: a system loaded with the Python Standalone Application
        :param wave_idx: Zemax wavelength number
        :param config: Zemax configuration number
        :param field: a list of Zemax field point numbers
        :param surface: Zemax surface number
        :param N_rays: Number of rays along the P_x [-1, 1] line. Total rays = N_rays ^ 2
        :return:
        """

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        N_fields = sysField.NumberOfFields

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        # Pupil Rays
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx**2 + pyy**2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]

        XY = np.empty((len(field), N_rays ** 2, 2))
        XY[:] = np.NaN          # Fill it with Nan so we can discard the rays outside the pupil

        # Loop over each Field computing the Spot Diagram
        for j, field_idx in enumerate(field):

            fx, fy = sysField.GetField(field_idx).X, sysField.GetField(field_idx).Y
            hx, hy = fx / r_max, fy / r_max      # Normalized field coordinates (hx, hy)

            raytrace = system.Tools.OpenBatchRayTrace()
            # remember to specify the surface to which you are tracing!
            normUnPolData = raytrace.CreateNormUnpol(N_rays_inside, constants.RaysType_Real, surface)

            for (p_x, p_y) in zip(px, py):
                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

            # Run the RayTrace for the whole Slice
            CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
            normUnPolData.StartReadingResults()
            for i in range(N_rays_inside):
                output = normUnPolData.ReadNextResult()
                if output[2] == 0 and output[3] == 0:
                    XY[j, i, 0] = output[4]
                    XY[j, i, 1] = output[5]

            normUnPolData.ClearData()
            CastTo(raytrace, 'ISystemTool').Close()

        # It is extremely important to output a LIST of arrays even if you only need 1 array
        # Otherwise, if you only have 1 result array, the reading of the results in run_analysis
        # will loop over the first axis and Python will broadcast incorrectly

        return [XY]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, field=(1,), N_rays=20, plot_mode='Fields'):


        N_fields = len(field)
        results_names = ['XY']
        results_shapes = [(N_fields, N_rays**2, 2)]             # [N_fields, N_rays^2, (x,y)]

        # read the file options
        file_list, settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        for zemax_file in file_list:

            file_name = zemax_file.split('.')[0]

            list_results = self.run_analysis(analysis_function=self.analysis_function_spot_diagram,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, field=field, N_rays=N_rays)

            xy, wavelengths = list_results

            if plot_mode == 'Fields':
                self.plot_fields(xy, wavelengths, wavelength_idx, configuration_idx, field,
                                 surface, file_name, results_path)

            return

    def plot_fields(self, xy, wavelengths, wavelength_idx, configuration_idx, fields, surface, file_name, result_path,
                    scale=5, cmap=cm.Reds):

        file_dir = os.path.join(results_path, file_name)
        spot_dir = os.path.join(file_dir, 'SPOT_DIAGRAM_FIELDS')
        if not os.path.exists(spot_dir):
            os.mkdir(spot_dir)           # If not, create the directory to store results

        # xy has shape [N_waves, N_configs, N_fields, N_rays^2, 2]
        N_waves, N_configs = xy.shape[0], xy.shape[1]
        N_fields = len(fields)
        colors = cmap(np.linspace(0.5, 1.0, N_waves))

        if wavelength_idx is None:
            wavelength_idx = np.arange(1, N_waves + 1)
        if surface is None:
            surface = '_IMG'

        for i in range(N_waves):
            wave = wavelengths[i]
            wave_idx = wavelength_idx[i]
            for j in range(N_configs):
                config = configuration_idx[j]

                fig_name = file_name + '_SPOTDIAG_SURF' + str(surface) + '_WAVE%d' % wave_idx + '_CONFIG%d' % config
                data = xy[i, j]

                deltax = max([np.nanmax(np.abs(data[k, :, 0] - np.nanmean(data[k, :, 0]))) for k in range(N_fields)])
                deltay = max([np.nanmax(np.abs(data[k, :, 1] - np.nanmean(data[k, :, 1]))) for k in range(N_fields)])
                xmax = 1.2 * max(deltax, deltay)        # some oversizing
                fig, axes = plt.subplots(1, N_fields, figsize=(scale*N_fields, scale))
                alpha = np.linspace(0.5, 1.0, N_fields)
                for k in range(N_fields):
                    ax = axes[k]
                    ax.set_aspect('equal')
                    x, y = data[k, :, 0], data[k, :, 1]
                    mx, my = np.nanmean(x), np.nanmean(y)
                    dx, dy = x - mx, y - my
                    ax.scatter(dx, dy, s=3, color=colors[i], alpha=alpha[k])
                    ax.set_title('Field #%d ($\lambda$: %.4f $\mu$m, cfg: %d)' % (fields[k], wave, config))
                    ax.set_xlabel(r'X [$\mu$m]')
                    ax.set_ylabel(r'Y [$\mu$m]')
                    ax.set_xlim([-xmax, xmax])
                    ax.set_ylim([-xmax, xmax])
                    if k > 0:
                        ax.get_yaxis().set_visible(False)

                if os.path.isfile(os.path.join(spot_dir, fig_name)):
                    os.remove(os.path.join(spot_dir, fig_name))
                fig.savefig(os.path.join(spot_dir, fig_name))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close('all')

        return


class SpotDiagramDetector(AnalysisGeneric):

    def analysis_functions_spots(self, system, wave_idx, config, surface, N_rays, reference='ChiefRay'):

        # colors = cm.Reds(np.linspace(0.5, 1, 23))
        # color = colors[wave_idx - 1]


        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields

        N_fields = sysField.NumberOfFields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        # Pupil Rays
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx**2 + pyy**2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]

        XY = np.empty((1, N_rays_inside + 1, 2))     # Rays inside plus 1 for the chief ray extra

        obj_xy = np.zeros((1, 2))        # (fx, fy) coordinates
        foc_xy = np.empty((1, 2))        # raytrace results of the Centroid

        if config == 1:
            print("\nTracing %d rays to calculate FWHM PSF" % (N_rays_inside))

        # fig, axes = plt.subplots(1, N_fields)

        # Loop over each Field computing the Spot Diagram


        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        hx, hy = fx / r_max, fy / r_max      # Normalized field coordinates (hx, hy)

        j = 0
        obj_xy[j, :] = [fx, fy]

        raytrace = system.Tools.OpenBatchRayTrace()
        # remember to specify the surface to which you are tracing!
        normUnPolData = raytrace.CreateNormUnpol(N_rays_inside + 1, constants.RaysType_Real, surface)

        # Add the Chief Ray
        normUnPolData.AddRay(wave_idx, hx, hy, 0.0, 0.0, constants.OPDMode_None)

        for (p_x, p_y) in zip(px, py):
            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        for i in range(N_rays_inside + 1):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                XY[j, i, 0] = output[4]
                XY[j, i, 1] = output[5]

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        x, y = XY[j, :, 0], XY[j, :, 1]
        chief_x, chief_y = x[0], y[0]           # We added the Chief Ray first to the BatchTrace
        cent_x, cent_y = np.mean(x), np.mean(y)

        # Select what point will be used as Reference for the FWHM
        if reference == "Centroid":
            ref_x, ref_y = cent_x, cent_y
        elif reference == "ChiefRay":
            ref_x, ref_y = chief_x, chief_y
        else:
            raise ValueError("reference should be 'ChiefRay' or 'Centroid'")

        # Add the Reference Point to the Focal Plane Raytrace results
        foc_xy[j, :] = [ref_x, ref_y]

        # # Calculate the contours
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

        return [XY, obj_xy, foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, N_rays=40, reference='ChiefRay'):
        """

        """

        # We need to know how many rays are inside the pupil
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx ** 2 + pyy ** 2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['XY', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(1, N_rays_inside + 1, 2), (1, 2), (1, 2)]

        # read the file options
        file_list, settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        for zemax_file in file_list:

            list_results = self.run_analysis(analysis_function=self.analysis_functions_spots,
                                             files_dir=files_dir,zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, N_rays=N_rays, reference=reference)

            return list_results



class RMSSpotAnalysis(AnalysisGeneric):

    @staticmethod
    def analysis_function_rms_spot(system, wave_idx, config, surface, field,
                                   reference='Centroid', mode='RMS', ray_density=10):

        N_fields = len(field)
        spots = np.zeros(N_fields)

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Create Spot Diagram analysis
        spot = system.Analyses.New_Analysis(constants.AnalysisIDM_StandardSpot)
        spot_setting = spot.GetSettings()
        baseSetting = CastTo(spot_setting, 'IAS_Spot')
        baseSetting.Field.SetFieldNumber(0)                     # 0 means All Field Points
        baseSetting.Wavelength.SetWavelengthNumber(wave_idx)    # set the Wavelength
        baseSetting.RayDensity = ray_density                    # set the Ray Density

        if reference == 'Centroid':
            baseSetting.ReferTo = constants.ReferTo_Centroid
        elif reference == 'ChiefRay':
            baseSetting.ReferTo = constants.ReferTo_ChiefRay

        base = CastTo(spot, 'IA_')
        base.ApplyAndWaitForCompletion()

        # Get Results
        spot_results = base.GetResults()
        for i, f in enumerate(field):
            if mode == 'RMS':
                spots[i] = spot_results.SpotData.GetRMSSpotSizeFor(f, wave_idx)
            elif mode == 'GEO':
                spots[i] = spot_results.SpotData.GetGeoSpotSizeFor(f, wave_idx)

        spot.Close()

        return [spots]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, field=(1,),
                        reference='Centroid', mode='RMS', ray_density=6, save_txt=False):

        N_fields = len(field)
        results_names = ['RMS_SPOT']
        results_shapes = [(N_fields,)]

        # read the file options
        file_list, settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        list_spot = []
        list_waves = []
        for zemax_file in file_list:

            list_results = self.run_analysis(analysis_function=self.analysis_function_rms_spot,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, field=field, reference=reference, mode=mode, ray_density=ray_density)

            rms_spot, wavelengths = list_results
            list_spot.append(rms_spot)
            list_waves.append(wavelengths)

            # mean_field_spot = np.mean(spots, axis=0)
            # plt.figure()
            # plt.imshow(mean_field_spot, origin='lower', cmap='Reds')
            # plt.colorbar()
            # plt.ylabel(r'Slice #')
            # plt.xlabel(r'Wavelength [$\mu$m]')
            #
            # mean_spot_wave = np.mean(spots, axis=(0, 1))
            #
            # plt.figure()
            # plt.plot(wavelengths, mean_spot_wave)
            # plt.xlabel(r'Wavelength [$\mu$m]')
            # plt.show()
        self.plot_results(list_spot, list_waves, file_list, results_path, gratings=files_opt['grating'])

        return list_spot

    def plot_results(self, list_spot, list_waves, file_list, results_path, gratings):


        # WARNING: this will only work if you have ran the code for a single SCALE and IFU
        file_no_spec = file_list[0].split('_SPEC')[0]

        file_dir = os.path.join(results_path, file_no_spec)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        spot_dir = os.path.join(file_dir, 'RMS_SPOT_WAVELENGTH')
        if not os.path.exists(spot_dir):
            os.mkdir(spot_dir)           # If not, create the directory to store results

        N_gratings = len(gratings)
        colors = cm.jet(np.linspace(0, 1, N_gratings))
        plt.figure()
        for k in range(N_gratings):
            spots, waves = list_spot[k], list_waves[k]
            mean_spots = np.mean(spots, axis=(1, 2))
            plt.plot(waves, mean_spots, color=colors[k], label=gratings[k])
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(r'RMS Spot [$\mu$m]')
        plt.ylim(bottom=0)
        plt.legend(title=r'Grating', loc=3, ncol=N_gratings//3)
        figname = file_no_spec + '_RMS_SPOT_WAVELENGTH'
        plt.title(figname)
        plt.savefig(os.path.join(spot_dir, figname))

        return

class EnsquaredEnergyCustomAnalysis(AnalysisGeneric):
    """
    Ensquared Energy calculations done by
    raytracing and counting rays inside a given box
    so that we can apply it to HARMONI where the spaxels
    are rectangular
    """
    def __init__(self, zosapi):
        super().__init__(zosapi=zosapi)

        # Define the Spaxel Sizes [um] for each Spaxel Scale at each Focal Plane
        self.spaxel_sizes = {}
        self.spaxel_sizes["60x30"] = {"FPRS": [97.656, 195.313],
                                      "PO": [195.313, 390.625],
                                      "IFU": [65, 130],
                                      "SPEC": [15, 30]}
        self.spaxel_sizes["20x20"] = {"FPRS": [65.104, 65.104],
                                      "PO": [195.313, 390.625],
                                      "IFU": [65, 130],
                                      "SPEC": [15, 30]}
        self.spaxel_sizes["10x10"] = {"FPRS": [35.552, 35.552],
                                      "PO": [195.313, 390.625],
                                      "IFU": [65, 130],
                                      "SPEC": [15, 30]}
        self.spaxel_sizes["4x4"] = {"FPRS": [13.021, 13.021],
                                      "PO": [195.313, 390.625],
                                      "IFU": [65, 130],
                                      "SPEC": [15, 30]}
        return

    def analysis_function_ensquared_energy(self, system, wave_idx, config, surface, scale, focal_plane, N_rays):

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields

        N_fields = sysField.NumberOfFields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        fy_mean = np.mean([sysField.GetField(i).Y for i in np.arange(1, N_fields + 1)])
        fx_mean = 1/2 * np.sum([sysField.GetField(i).X for i in np.arange(1, N_fields + 1)])

        ## Add some extra field to get a better sampling
        sysField.AddField(fx_mean, fy_mean, 1)
        sysField.AddField(fx_mean / 3.0, fy_mean, 1)
        N_fields = sysField.NumberOfFields


        # Pupil Rays
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx**2 + pyy**2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]

        XY = np.empty((N_fields, N_rays ** 2, 2))
        XY[:] = np.NaN          # Fill it with Nan so we can discard the rays outside the pupil

        EE = np.zeros(N_fields)
        obj_xy = np.zeros((N_fields, 2))        # (fx, fy) coordinates
        foc_xy = np.empty((N_fields, 2))        # raytrace results of the Centroid

        # Loop over each Field computing the Spot Diagram
        for j in range(N_fields):

            fx, fy = sysField.GetField(j + 1).X, sysField.GetField(j + 1).Y
            hx, hy = fx / r_max, fy / r_max      # Normalized field coordinates (hx, hy)

            obj_xy[j, :] = [fx, fy]

            raytrace = system.Tools.OpenBatchRayTrace()
            # remember to specify the surface to which you are tracing!
            normUnPolData = raytrace.CreateNormUnpol(N_rays_inside, constants.RaysType_Real, surface)

            for (p_x, p_y) in zip(px, py):
                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

            # Run the RayTrace for the whole Slice
            CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
            normUnPolData.StartReadingResults()
            for i in range(N_rays_inside):
                output = normUnPolData.ReadNextResult()
                if output[2] == 0 and output[3] == 0:
                    XY[j, i, 0] = output[4]
                    XY[j, i, 1] = output[5]

            normUnPolData.ClearData()
            CastTo(raytrace, 'ISystemTool').Close()

            # Calculate Centroid
            cent_x, cent_y = np.nanmean(XY[j, :, 0]), np.nanmean(XY[j, :, 1])
            foc_xy[j, :] = [cent_x, cent_y]

            # Get the size of the spaxel
            spax_x, spax_y = self.spaxel_sizes[scale][focal_plane]      # in microns
            spax_x /= 1000
            spax_y /= 1000

            xmin, xmax = cent_x - spax_x, cent_x + spax_x
            ymin, ymax = cent_y - spax_y, cent_y + spax_y

            # Find out how many rays we need to ignore bc they are outside the pupil

            N_good = np.isfinite(XY[j, :, 0]).sum()
            inside = 0
            for i, is_finite in enumerate(np.isfinite(XY[j, :, 0])):
                # print(i, is_nan)
                if is_finite == True:      # a proper number
                    x, y = XY[j, i, 0], XY[j, i, 1]
                    #TODO: fix the size of the spaxel
                    # print(i)
                    if xmin < x < xmax and ymin < y < ymax:     # Inside
                        inside += 1
            EE[j] = inside / N_good

            # fig, ax1 = plt.subplots(1, 1)
            # ax1.scatter(XY[j, :, 0], XY[j, :, 1], s=4)
            # square = Rectangle((cent_x - spax_x/2, cent_y - spax_y/2), spax_x, spax_y, linestyle='--', fill=None, color='black')
            # ax1.add_patch(square)
            # # ax1.set_xlim([xmin, xmax])
            # # ax1.set_ylim([ymin, ymax])
            # plt.title("Field #%d (%.3f, %.3f)" % (j + 1, fx, fy))

        # print(EE.shape)

        # Remember to remove the extra fields otherwise they pile up
        sysField.RemoveField(N_fields)
        sysField.RemoveField(N_fields - 1)


        return [EE, obj_xy, foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, scale="60x30", focal_plane="PO", N_rays=30):
        """


        :return:
        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['EE', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(5,), (5, 2), (5, 2)]

        # read the file options
        file_list, settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        for zemax_file in file_list:

            list_results = self.run_analysis(analysis_function=self.analysis_function_ensquared_energy,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, scale=scale, focal_plane=focal_plane, N_rays=N_rays)

            ensq_ener, obj_xy, foc_xy, wavelengths = list_results

            min_ee_wave = np.min(ensq_ener, axis=(1, 2))
            max_ee_wave = np.max(ensq_ener, axis=(1, 2))
            plt.figure()
            plt.plot(wavelengths, min_ee_wave, label='Min')
            plt.plot(wavelengths, max_ee_wave, label='Max')
            plt.legend()
            plt.xlabel(r'Wavelength [$\mu$m]')
            plt.ylim([0.90, 1.1])
            plt.ylabel(r'Ensquared Energy [ ]')

            print("\nEnsquared Energy Results")
            min_ee = np.min(ensq_ener)
            max_ee = np.max(ensq_ener)
            print("Min: %.3f , Max: %.3f" % (min_ee, max_ee))

            N_waves = len(wavelengths)
            if wavelength_idx is None:
                wave_idx = np.arange(1, N_waves + 1)
            else:
                wave_idx = wavelength_idx

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            results_dir = os.path.join(results_path, file_name)
            surface_number = str(surface) if surface is not None else '_IMG'

            #
            # ### Object Space Plots
            # for k in range(N_waves):
            #     x, y = obj_xy[k, :, :, 0].flatten(), obj_xy[k, :, :, 1].flatten()
            #     z = ensq_ener[k].flatten()
            #
            #     plt.figure()
            #     plt.tricontourf(x, y, z, cmap='Blues')
            #     plt.clim(vmin=0.5, vmax=np.max(z))
            #     plt.plot(x, y, 'ko', ms=2)
            #     plt.xlabel(r'Object $f_x$ [mm]')
            #     plt.ylabel(r'Object $f_y$ [mm]')
            #     plt.colorbar(orientation='horizontal')
            #     plt.xlim([-1.05 * x.min(), 1.05 * x.min()])
            #
            #     fig_name = file_name + '_OBJ_SURF_' + surface_number + '_ENSQ_ENER_WAVE%d' % (wave_idx[k])
            #     plt.title(fig_name)
            #
            #     plt.axes().set_aspect('equal')
            #     if os.path.isfile(os.path.join(results_dir, fig_name)):
            #         os.remove(os.path.join(results_dir, fig_name))
            #     plt.savefig(os.path.join(results_dir, fig_name))
            #     plt.show(block=False)
            #     plt.pause(0.5)
            #     plt.close()
            #
            # if focal_plane == "SPEC":
            #     ### Focal Plane Plots
            #     x, y = foc_xy[:, :, :, 0].flatten(), foc_xy[:, :, :, 1].flatten()
            #     z = ensq_ener.flatten()
            #
            #     plt.figure()
            #     img = plt.tricontourf(x, y, z, cmap='Blues')
            #     img.set_clim(vmin=0.5, vmax=1.0)
            #     plt.scatter(x, y, s=2, color='white')
            #     plt.xlabel(r'Focal Plane X [mm]')
            #     plt.ylabel(r'Focal Plane Y [mm]')
            #     plt.colorbar(img, orientation='horizontal')
            #     plt.xlim([-1.05 * x.min(), 1.05 * x.min()])
            #     plt.ylim([-1.05 * y.min(), 1.05 * y.min()])
            #     fig_name = file_name + '_FOC' + surface_number + '_ENSQ_ENER'
            #     plt.title(fig_name)
            #     plt.axes().set_aspect('equal')
            #     if os.path.isfile(os.path.join(results_dir, fig_name)):
            #         os.remove(os.path.join(results_dir, fig_name))
            #     plt.savefig(os.path.join(results_dir, fig_name))
            #     plt.show(block=False)
            #     plt.pause(0.5)
            #     plt.close()

        return list_results



class EnsquaredEnergyAnalysis(AnalysisGeneric):
    """
    Ensquared Energy using the Zemax operand GENF
    Only works for SQUARE cases
    """

    @staticmethod
    def analysis_function_ensquared_energy(system, wave_idx, config, surface, distance, sampling):

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields

        N_fields = sysField.NumberOfFields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        fy_mean = np.mean([sysField.GetField(i).Y for i in np.arange(1, N_fields + 1)])
        fx_mean = 1/2 * np.sum([sysField.GetField(i).X for i in np.arange(1, N_fields + 1)])

        ## Add some extra field to get a better sampling
        sysField.AddField(fx_mean, fy_mean, 1)
        sysField.AddField(fx_mean / 3.0, fy_mean, 1)
        N_fields = sysField.NumberOfFields
        # new_field = sysField.GetField(N_fields)
        # print("%.3f, %.3f" % (new_field.X, new_field.Y))
        # new_field = sysField.GetField(N_fields-1)
        # print("%.3f, %.3f" % (new_field.X, new_field.Y))
        # print(N_fields)

        # Run the RayTrace for each Field Point
        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(N_fields, constants.RaysType_Real, surface)

        obj_xy = np.zeros((N_fields, 2))        # (fx, fy) coordinates
        foc_xy = np.empty((N_fields, 2))        # raytrace results
        ensq_ener = np.empty(N_fields)

        theMFE = system.MFE
        # clear current merit function
        nops = theMFE.NumberOfOperands
        theMFE.RemoveOperandsAt(1, nops)
        # build merit function
        op = theMFE.GetOperandAt(1)
        op.ChangeType(constants.MeritOperandType_CONF)
        op.GetOperandCell(constants.MeritColumn_Param1).Value = config

        for i in range(N_fields):       # Loop over the available Field Points

            fx, fy = sysField.GetField(i + 1).X, sysField.GetField(i + 1).Y
            hx, hy = fx / r_max, fy / r_max     # Normalized field coordinates (hx, hy)
            obj_xy[i, :] = [fx, fy]

            op = theMFE.AddOperand()
            op.ChangeType(constants.MeritOperandType_GENF)
            op.GetOperandCell(constants.MeritColumn_Param1).Value = sampling    # Pupil Sampling {1: 32x32, 2: 64x64 ...]
            op.GetOperandCell(constants.MeritColumn_Param2).Value = wave_idx    # Wave
            op.GetOperandCell(constants.MeritColumn_Param3).Value = i + 1       # Field
            op.GetOperandCell(constants.MeritColumn_Param4).Value = distance    # Distance [microns]
            op.GetOperandCell(constants.MeritColumn_Param5).Value = 4           # Type {4: Ensquared}
            op.GetOperandCell(constants.MeritColumn_Param6).Value = 0           # Reference {0: Chief ray, 1: Centroid}
            op.GetOperandCell(constants.MeritColumn_Weight).Value = 0

            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, hx, hy, 0, 0, constants.OPDMode_None)

        # update merit function
        theMFE.CalculateMeritFunction()
        for irow in range(2, theMFE.NumberOfOperands + 1):
            if op.Type == constants.MeritOperandType_GENF:      # Sanity check that we are looking at the correct Type
                op = theMFE.GetOperandAt(irow)
                # print(op.Value)
                ensq_ener[irow - 2] = op.Value
        # print("_")

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        for i in range(N_fields):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                foc_xy[i, 0] = output[4]
                foc_xy[i, 1] = output[5]

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # Remove that extra Field we have added, otherwise they pile up
        sysField.RemoveField(N_fields)
        sysField.RemoveField(N_fields - 1)

        return [ensq_ener, obj_xy, foc_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, distance=1.0, sampling=4):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_rms_wfe


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param surface: Zemax Surface number at which the analysis will be computed

        :return:
        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['EE_WFE', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(5,), (5, 2), (5, 2)]

        # read the file options
        file_list, settings = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        for zemax_file in file_list:

            list_results = self.run_analysis(analysis_function=self.analysis_function_ensquared_energy,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, distance=distance, sampling=sampling)

            ensq_ener, obj_xy, foc_xy, wavelengths = list_results

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            results_dir = os.path.join(results_path, file_name)
            surface_number = str(surface) if surface is not None else '_IMG'

            for k in range(1):
                x, y = obj_xy[k, :, :, 0].flatten(), obj_xy[k, :, :, 1].flatten()
                z = ensq_ener[k].flatten()

                # ngridx = 100
                # xi = np.linspace(x.min(), x.max(), ngridx)
                # yi = np.linspace(y.min(), y.max(), ngridx)
                # zi = mlab.griddata(x, y, z, xi, yi, interp='linear')

                plt.figure()
                # triang = tri.Triangulation(x, y)
                # plt.tricontour(x, y, z, 10, linewidths=0.5, colors='k')
                plt.tricontourf(x, y, z, 25,
                                cmap='Blues')
                plt.clim(vmin=np.min(z), vmax=np.max(z))
                plt.plot(x, y, 'ko', ms=2)
                plt.xlabel(r'X [mm]')
                plt.ylabel(r'Y [mm]')
                fig_name = file_name + '_SURF_' + surface_number + '_ENSQ_ENER_WAVE%d' % (wavelength_idx[k])
                plt.title(fig_name)
                plt.colorbar(orientation='horizontal')
                plt.xlim([-1.05 * x.min(), 1.05 * x.min()])

                plt.axes().set_aspect('equal')
                if os.path.isfile(os.path.join(results_dir, fig_name)):
                    os.remove(os.path.join(results_dir, fig_name))
                plt.savefig(os.path.join(results_dir, fig_name))
                plt.show(block=False)
                plt.pause(0.5)
                plt.close()

        return list_results



class GeometricFWHM_PSF_Analysis(AnalysisGeneric):
    """
    Calculate the Geometric FWHM of the PSF
    by computing the spot diagrams, getting the contour plots
    and estimating the diameter of the 50% contour line
    """

    def analysis_function_geofwhm_psf(self, system, wave_idx, config, surface, N_rays, N_points,
                                      PSF_window, reference='Centroid'):


        # print("Config: ", config)
        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields

        N_fields = sysField.NumberOfFields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        # fy_mean = np.mean([sysField.GetField(i).Y for i in np.arange(1, N_fields + 1)])
        # fx_mean = 1/2 * np.sum([sysField.GetField(i).X for i in np.arange(1, N_fields + 1)])
        #
        # ## Add some extra field to get a better sampling
        # sysField.AddField(fx_mean, fy_mean, 1)
        # sysField.AddField(fx_mean / 3.0, fy_mean, 1)
        # N_fields = sysField.NumberOfFields

        # Pupil Rays
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx**2 + pyy**2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]
        diam = 39       # meters
        freq = diam / N_rays       # pupil plane frequency

        XY = np.empty((N_fields, N_rays_inside + 1, 2))     # Rays inside plus 1 for the chief ray extra

        FWHM = np.zeros((N_fields, 3))          # [mean_radius, m_radiusX, m_radiusY]
        obj_xy = np.zeros((N_fields, 2))        # (fx, fy) coordinates
        foc_xy = np.empty((N_fields, 2))        # raytrace results of the Centroid
        PSF_cube = np.empty((N_fields, N_points, N_points))

        if config == 1:
            print("\nTracing %d rays to calculate FWHM PSF" % (N_rays_inside))

        # fig, axes = plt.subplots(1, N_fields)

        # Loop over each Field computing the Spot Diagram
        for j in range(N_fields):

            fx, fy = sysField.GetField(j + 1).X, sysField.GetField(j + 1).Y
            hx, hy = fx / r_max, fy / r_max      # Normalized field coordinates (hx, hy)

            obj_xy[j, :] = [fx, fy]

            raytrace = system.Tools.OpenBatchRayTrace()
            # remember to specify the surface to which you are tracing!
            normUnPolData = raytrace.CreateNormUnpol(N_rays_inside + 1, constants.RaysType_Real, surface)

            # Add the Chief Ray
            normUnPolData.AddRay(wave_idx, hx, hy, 0.0, 0.0, constants.OPDMode_None)

            for (p_x, p_y) in zip(px, py):
                # Add the ray to the RayTrace
                normUnPolData.AddRay(wave_idx, hx, hy, p_x, p_y, constants.OPDMode_None)

            # Run the RayTrace for the whole Slice
            CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
            normUnPolData.StartReadingResults()
            for i in range(N_rays_inside + 1):
                output = normUnPolData.ReadNextResult()
                if output[2] == 0 and output[3] == 0:
                    XY[j, i, 0] = output[4]
                    XY[j, i, 1] = output[5]

            normUnPolData.ClearData()
            CastTo(raytrace, 'ISystemTool').Close()

            # Calculate the contours
            x, y = XY[j, :, 0], XY[j, :, 1]
            xy = XY[j]

            chief_x, chief_y = x[0], y[0]           # We added the Chief Ray first to the BatchTrace
            cent_x, cent_y = np.mean(x), np.mean(y)

            # Select what point will be used as Reference for the FWHM
            if reference == "Centroid":
                ref_x, ref_y = cent_x, cent_y
            elif reference == "ChiefRay":
                ref_x, ref_y = chief_x, chief_y
            else:
                raise ValueError("reference should be 'ChiefRay' or 'Centroid'")

            # Add the Reference Point to the Focal Plane Raytrace results
            foc_xy[j, :] = [ref_x, ref_y]

            # KDE with scikit
            std_x, std_y = np.std(x), np.std(y)
            std = max(std_x, std_y)
            bandwidth = min(std_x, std_y)

            kde = KernelDensity(kernel='gaussian', bandwidth=0.75*bandwidth).fit(xy)

            # define a grid to compute the PSF
            # xmin, xmax = cent_x - 3 * std, cent_x + 3 * std
            # ymin, ymax = cent_y - 3 * std, cent_y + 3 * std
            xmin, xmax = cent_x - PSF_window/2/1000, cent_x + PSF_window/2/1000
            ymin, ymax = cent_y - PSF_window/2/1000, cent_y + PSF_window/2/1000
            x_grid = np.linspace(xmin, xmax, N_points)
            y_grid = np.linspace(ymin, ymax, N_points)
            xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
            xy_grid = np.vstack([xx_grid.ravel(), yy_grid.ravel()]).T
            # print(xy_grid.shape)
            log_scores = kde.score_samples(xy_grid)

            psf = np.exp(log_scores)
            psf /= np.max(psf)
            psf = psf.reshape(xx_grid.shape)
            PSF_cube[j] = psf

            # ax = axes[j]
            # img = ax.imshow(psf, extent=[xmin, xmax, ymin, ymax], cmap='bwr', origin='lower')
            # # ax.scatter(x, y, s=3, color='black', alpha=0.5)
            # plt.colorbar(img, ax=ax, orientation='horizontal')
            # ax.set_xlim([xmin, xmax])
            # ax.set_ylim([ymin, ymax])

            # FWHM[j, :] = [-99, fwhm_x, fwhm_y]        # Store the results

        # plt.show()
        #
        # plt.show(block=False)
        # plt.pause(0.05)
        # plt.close(fig)

        # # Remember to remove the extra fields otherwise they pile up
        # sysField.RemoveField(N_fields)
        # sysField.RemoveField(N_fields - 1)

        return [FWHM, PSF_cube, obj_xy, foc_xy]


    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, N_rays=40, N_points=100, PSF_window=50, plots=False):
        """

        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['GEO_FWHM', 'PSF_CUBE', 'OBJ_XY', 'FOC_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(3, 3), (3, N_points, N_points), (3, 2), (3, 2)]

        # We need to know how many rays are inside the pupil
        px = np.linspace(-1, 1, N_rays, endpoint=True)
        pxx, pyy = np.meshgrid(px, px)
        pupil_mask = np.sqrt(pxx ** 2 + pyy ** 2) <= 1.0
        px, py = pxx[pupil_mask], pyy[pupil_mask]
        # How many rays are actually inside the pupil aperture?
        N_rays_inside = px.shape[0]

        metadata = {}
        metadata['N_rays Pupil'] = N_rays_inside
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx
        metadata['N_points PSF'] = N_points
        metadata['PSF window [microns]'] = PSF_window

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'], AO_modes=files_opt['AO_modes'],
                                           scales=files_opt['scales'], IFUs=files_opt['IFUs'], grating=files_opt['grating'])
        # print(settings)

        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_geofwhm_psf,
                                             files_dir=files_dir,zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, N_rays=N_rays, N_points=N_points, PSF_window=PSF_window)


            geo_fwhm, psf_cube, obj_xy, foc_xy, wavelengths = list_results

            settings['surface'] = 'IMG' if surface is None else surface

            file_name = zemax_file.split('.')[0]
            self.save_hdf5(analysis_name='FWHM_PSF', analysis_metadata=metadata, list_results=list_results,
                           results_names=results_names, file_name=file_name, file_settings=settings, results_dir=results_path)

            # N_waves = geo_fwhm.shape[0]
            #
            # if wavelength_idx is None:
            #     wave_idx = np.arange(1, N_waves + 1)
            # else:
            #     wave_idx = wavelength_idx
            #
            #
            # # Post-Processing the results
            # file_name = zemax_file.split('.')[0]
            # results_dir = os.path.join(results_path, file_name)
            # surface_number = str(surface) if surface is not None else '_IMG'
            # settings['surface'] = surface
            #
            # surface_codes = focal_planes[settings['system']][settings['scale']]
            # surface_name = list(surface_codes.keys())[list(surface_codes.values()).index(surface)]

            # # First thing is to create a separate folder within the results directory for this analysis
            # analysis_dir = os.path.join(results_dir, 'FWHM')
            # print("Analysis Results will be saved in folder: ", analysis_dir)
            # if not os.path.exists(analysis_dir):
            #     os.mkdir(analysis_dir)
            #
            # if surface == None:
            #
            #     x, y = foc_xy[:, :, :, 0].flatten(), foc_xy[:, :, :, 1].flatten()
            #     triang = tri.Triangulation(x, y)
            #     maxz = np.max(geo_fwhm)
            #
            #     fig1, axes = plt.subplots(1, 3)
            #     # ax1.set_aspect('equal')
            #     titles = [r'FWHM [$\mu$m]', r'FWHM_X [$\mu$m]', r'FWHM_Y [$\mu$m]']
            #     for k in range(3):
            #         ax = axes[k]
            #         tpc = ax.tripcolor(triang, geo_fwhm[:, :, :, k].flatten(), shading='flat', cmap='jet')
            #         tpc.set_clim(vmin=0, vmax=maxz)
            #         ax.scatter(x, y, color='black', s=2)
            #         plt.colorbar(tpc, orientation='horizontal', ax=ax)
            #         ax.set_xlabel(r'X [mm]')
            #         if k == 0:
            #             ax.set_ylabel(r'Y [mm]')
            #         ax.set_title(titles[k])
            #
            #     fig_name = file_name + '_FWHM_SURF_' + surface_name + '_DET'
            #     # plt.title(fig_name)
            #     if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            #         os.remove(os.path.join(analysis_dir, fig_name))
            #     plt.savefig(os.path.join(analysis_dir, fig_name))
            #
            # else:
            #     # fwhm is shape [N_waves, N_configs, N_fields, (meanR, XR, YR)]
            #     for j_wave in range(N_waves):
            #         x, y = foc_xy[j_wave, :, :, 0].flatten(), foc_xy[j_wave, :, :, 1].flatten()
            #         triang = tri.Triangulation(x, y)
            #         maxz = np.max(geo_fwhm[j_wave])
            #
            #         fig1, axes = plt.subplots(1, 3)
            #         # ax1.set_aspect('equal')
            #         titles = [r'FWHM [$\mu$m]', r'FWHM_X [$\mu$m]', r'FWHM_Y [$\mu$m]']
            #         for k in range(3):
            #             ax = axes[k]
            #             tpc = ax.tripcolor(triang, geo_fwhm[j_wave, :, :, k].flatten(), shading='flat', cmap='jet')
            #             tpc.set_clim(vmin=0, vmax=maxz)
            #             ax.scatter(x, y, color='black', s=2)
            #             plt.colorbar(tpc, orientation='horizontal', ax=ax)
            #             ax.set_xlabel(r'X [mm]')
            #             if k == 0:
            #                 ax.set_ylabel(r'Y [mm]')
            #             ax.set_title(titles[k])
            #
            #         fig_name = file_name + '_FWHM_SURF_' + surface_name + '_FOCAL_WAVE%d' % wave_idx[j_wave]
            #         # plt.title(fig_name)
            #         if os.path.isfile(os.path.join(analysis_dir, fig_name)):
            #             os.remove(os.path.join(analysis_dir, fig_name))
            #         plt.savefig(os.path.join(analysis_dir, fig_name))
            #     #     plt.show(block=False)
            #     #     plt.pause(0.5)
            #     #     plt.close()

        return list_results

class RaytraceAnalysis(AnalysisGeneric):
    """
    Raytrace analysis at any surface
    """

    @staticmethod
    def analysis_function_raytrace(system, wave_idx, config, rays_per_slice, surface):

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial
        N_fields = sysField.NumberOfFields

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])
        # TODO: make this robust
        # Watch Out! This assumes that Field #1 is at the edge of the slice and that Field #3 is at the other edge
        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(3).X, sysField.GetField(3).Y

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / r_max, fx_max / r_max
        hy_min, hy_max = fy_min / r_max, fy_max / r_max

        hx = np.linspace(hx_min, hx_max, rays_per_slice)
        hy = np.linspace(hy_min, hy_max, rays_per_slice)

        # The Field coordinates for the Object
        obj_xy = r_max * np.array([hx, hy]).T
        foc_xy = np.empty((rays_per_slice, 2))
        global_xy = np.empty((rays_per_slice, 2))
        local_xyz = np.empty((rays_per_slice, 3))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(rays_per_slice, constants.RaysType_Real, surface)

        # Loop over all Rays in the Slice
        for i, (h_x, h_y) in enumerate(zip(hx, hy)):
            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        for i in range(rays_per_slice):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                local_xyz[i, 0] = output[4]
                local_xyz[i, 1] = output[5]
                local_xyz[i, 2] = output[6]

                # Local Focal X Y
                foc_xy[i, 0] = output[4]
                foc_xy[i, 1] = output[5]

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # Get the transformation from Local to Global Coordinates
        global_mat = system.LDE.GetGlobalMatrix(surface)
        R11, R12, R13, R21, R22, R23, R31, R32, R33, X0, Y0, Z0 = global_mat[1:]
        global_matrix = np.array([[R11, R12, R13],
                                  [R21, R22, R23],
                                  [R31, R32, R33]])
        offset = np.array([X0, Y0, Z0])

        # Transform from Local to Global and only save X and Y
        global_xyz = (np.dot(global_matrix, local_xyz.T)).T + offset
        global_xy = global_xyz[:, :2]

        return [obj_xy, foc_xy, global_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, rays_per_slice=3):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_raytrace


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param surface: Zemax Surface number at which the analysis will be computed
        :param rays_per_slice: number rays to trace per slice
        :return:
        """

        results_names = ['OBJ_XY', 'FOC_XY', 'GLOB_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(rays_per_slice, 2), (rays_per_slice, 2), (rays_per_slice, 2)]

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_raytrace,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, rays_per_slice=rays_per_slice)

            obj_xy, foc_xy, global_xy, wavelengths = list_results

            # For each file we save the list of results
            # we could use that in a Monte Carlo analysis for example
            results.append(list_results)

        return results


class EnsquaredDetector(AnalysisGeneric):

    @staticmethod
    def analysis_function_ensquared(system, wave_idx, config, surface, size, N_rays):

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        N_fields = sysField.NumberOfFields

        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])

        # Get the Field Point at the centre of the Slice
        fx, fy = sysField.GetField(2).X, sysField.GetField(2).Y
        # Trace a Square around the centre point
        x_min, x_max = fx - size/2, fx + size/2
        y_min, y_max = fy - size/2, fy + size/2
        x_up, y_up = np.linspace(x_min, x_max, N_rays), y_max * np.ones(N_rays)
        x_down, y_down = np.linspace(x_min, x_max, N_rays), y_min * np.ones(N_rays)
        x_left, y_left = x_min * np.ones(N_rays), np.linspace(y_min, y_max, N_rays)
        x_right, y_right = x_max * np.ones(N_rays), np.linspace(y_min, y_max, N_rays)
        x_mid, y_mid = fx * np.ones(N_rays), np.linspace(y_min, y_max, N_rays)

        x = np.concatenate([x_down, x_right, x_up, x_left, x_mid])
        y = np.concatenate([y_down, y_right, y_up, y_left, y_mid])
        hx, hy = x / r_max, y / r_max

        obj_xy = r_max * np.array([hx, hy]).T
        foc_xy = np.empty((5*N_rays, 2))
        local_xyz = np.empty((5*N_rays, 3))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(5*N_rays, constants.RaysType_Real, surface)

        # Loop over all Rays in the Slice
        for i, (h_x, h_y) in enumerate(zip(hx, hy)):
            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        checksum = 0
        for i in range(5 * N_rays):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                local_xyz[i, 0] = output[4]
                local_xyz[i, 1] = output[5]
                local_xyz[i, 2] = output[6]

                # Local Focal X Y
                foc_xy[i, 0] = output[4]
                foc_xy[i, 1] = output[5]
                checksum += 1
        print("Rays Traced: %d / %d" % (checksum, 5*N_rays))

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # Get the transformation from Local to Global Coordinates
        global_mat = system.LDE.GetGlobalMatrix(surface)
        R11, R12, R13, R21, R22, R23, R31, R32, R33, X0, Y0, Z0 = global_mat[1:]
        global_matrix = np.array([[R11, R12, R13],
                                  [R21, R22, R23],
                                  [R31, R32, R33]])
        offset = np.array([X0, Y0, Z0])

        # Transform from Local to Global and only save X and Y
        global_xyz = (np.dot(global_matrix, local_xyz.T)).T + offset
        global_xy = global_xyz[:, :2]

        return [obj_xy, foc_xy, global_xy]

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, size=1.0, N_rays=100):


        results_names = ['OBJ_XY', 'FOC_XY', 'GLOB_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(5*N_rays, 2), (5*N_rays, 2), (5*N_rays, 2)]

        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_ensquared,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, size=size, N_rays=N_rays)

            obj_xy, foc_xy, global_xy, wavelengths = list_results

            # For each file we save the list of results
            # we could use that in a Monte Carlo analysis for example
            results.append(list_results)

        return results



class RMS_WFE_Analysis(AnalysisGeneric):
    """
    Example of how to use Class Inheritance with the AnalysisGeneric object
    to calculate the RMS WFE error

    In this case the "analysis_function" is "analysis_function_rms_wfe"
    """

    @staticmethod
    def analysis_function_rms_wfe(system, wave_idx, config, spaxels_per_slice, surface):
        """
        Analysis function that calculates the RMS Wavefront Error [nm]
        for an Optical System, at a given Wavelength and Configuration / Slice
        at a specified Surface

        This function will return a list of arrays:
            - RMS_WFE: array of size [N_spaxels_per_slice] containing the RMS WFE for each spaxel of the given config
            - OBJ_XY: array of size [N_spaxels_per_slice, 2] containing the (XY) coordinates at the Object plane
            - FOC_XY: same as OBJ_XY but at the Focal plane defined by Surface

        :param system: Optical System that the ZOS API has loaded
        :param wave_idx: Zemax Wavelength Number to be used in the RMS calculation
        :param config: Zemax Configuration Number, the current Slice
        :param spaxels_per_slice: how many spaxels per slice to use for sampling the slice
        :param surface: Zemax surface number at which to evaluate the RMS WFE
        :param rings: Zemax parameter for the RMS WFE calculation
        :return:
        """

        # Set Current Configuration
        system.MCE.SetCurrentConfiguration(config)

        # Get the Field Points for that configuration
        sysField = system.SystemData.Fields
        # check that the field is normalized correctly
        if sysField.Normalization != constants.FieldNormalizationType_Radial:
            sysField.Normalization = constants.FieldNormalizationType_Radial

        N_fields = sysField.NumberOfFields
        wavelength = system.SystemData.Wavelengths.GetWavelength(wave_idx).Wavelength

        # Loop over the fields to get the Normalization Radius
        r_max = np.max([np.sqrt(sysField.GetField(i).X ** 2 +
                                sysField.GetField(i).Y ** 2) for i in np.arange(1, N_fields + 1)])
        # print(r_max)

        # TODO: make this robust
        # Watch Out! This assumes that Field #1 is at the edge of the slice and that Field #3 is at the other edge
        fx_min, fy_min = sysField.GetField(1).X, sysField.GetField(1).Y
        fx_max, fy_max = sysField.GetField(3).X, sysField.GetField(3).Y

        # Normalized field coordinates (hx, hy)
        hx_min, hx_max = fx_min / r_max, fx_max / r_max
        hy_min, hy_max = fy_min / r_max, fy_max / r_max

        # print("h_x : (%.3f, %.3f)" % (hx_min, hx_max))
        # print("h_y : (%.3f, %.3f)" % (hy_min, hy_max))

        hx = np.linspace(hx_min, hx_max, spaxels_per_slice)
        hy = np.linspace(hy_min, hy_max, spaxels_per_slice)

        # The Field coordinates for the Object
        obj_xy = r_max * np.array([hx, hy]).T
        RMS_WFE = np.empty(spaxels_per_slice)
        foc_xy = np.empty((spaxels_per_slice, 2))
        global_xy = np.empty((spaxels_per_slice, 2))
        local_xyz = np.empty((spaxels_per_slice, 3))

        raytrace = system.Tools.OpenBatchRayTrace()
        normUnPolData = raytrace.CreateNormUnpol(spaxels_per_slice, constants.RaysType_Real, surface)

        # Loop over all Spaxels in the Slice
        for i, (h_x, h_y) in enumerate(zip(hx, hy)):

            operand = constants.MeritOperandType_RWCE
            rms = system.MFE.GetOperandValue(operand, surface, wave_idx, h_x, h_y, 0.0, 0.0, 0.0, 0.0)
            RMS_WFE[i] = wavelength * 1e3 * rms         # We assume the Wavelength comes in Microns

            # Add the ray to the RayTrace
            normUnPolData.AddRay(wave_idx, h_x, h_y, 0, 0, constants.OPDMode_None)

        # Run the RayTrace for the whole Slice
        CastTo(raytrace, 'ISystemTool').RunAndWaitForCompletion()
        normUnPolData.StartReadingResults()
        for i in range(spaxels_per_slice):
            output = normUnPolData.ReadNextResult()
            if output[2] == 0 and output[3] == 0:
                local_xyz[i, 0] = output[4]
                local_xyz[i, 1] = output[5]
                local_xyz[i, 2] = output[6]

                # Local Focal X Y
                foc_xy[i, 0] = output[4]
                foc_xy[i, 1] = output[5]

        normUnPolData.ClearData()
        CastTo(raytrace, 'ISystemTool').Close()

        # Get the transformation from Local to Global Coordinates
        global_mat = system.LDE.GetGlobalMatrix(surface)
        R11, R12, R13, R21, R22, R23, R31, R32, R33, X0, Y0, Z0 = global_mat[1:]
        global_matrix = np.array([[R11, R12, R13],
                                  [R21, R22, R23],
                                  [R31, R32, R33]])
        offset = np.array([X0, Y0, Z0])

        # Transform from Local to Global and only save X and Y
        global_xyz = (np.dot(global_matrix, local_xyz.T)).T + offset
        global_xy = global_xyz[:, :2]

        return [RMS_WFE, obj_xy, foc_xy, global_xy]

    # The following functions depend on how you want to post-process your analysis results
    # For example, "loop_over_files" automatically runs the RMS WFE analysis across all Zemax files
    # and uses "flip_rms" to reorder the Configurations -> Slice #,
    # and "plot_RMS_WFE_maps" to save the figures

    def loop_over_files(self, files_dir, files_opt, results_path, wavelength_idx=None,
                        configuration_idx=None, surface=None, spaxels_per_slice=51,
                        plots=True):
        """
        Function that loops over a given set of E2E model Zemax files, running the analysis
        defined by self.analysis_function_rms_wfe


        :param files_dir: path where the E2E model files are stored
        :param files_opt: dictionary containing the info to create the list of zemax files we want to analyse
        :param results_path: path where we want to store the results
        :param wavelength_idx: list containing the Wavelength numbers we want to analyze. If None, we will use All
        :param configuration_idx: list containing the Configurations we want to analyze. If None, we will use All
        :param surface: Zemax Surface number at which the analysis will be computed
        :param spaxels_per_slice: number spaxels to sample each slice with
        :return:
        """

        # We want the result to produce as output: the RMS WFE array, and the RayTrace at both Object and Focal plane
        results_names = ['RMS_WFE', 'OBJ_XY', 'FOC_XY', 'GLOB_XY']
        # we need to give the shapes of each array to self.run_analysis
        results_shapes = [(spaxels_per_slice,), (spaxels_per_slice, 2), (spaxels_per_slice, 2), (spaxels_per_slice, 2)]

        metadata = {}
        metadata['Spaxels per slice'] = spaxels_per_slice
        metadata['Configurations'] = 'All' if configuration_idx is None else configuration_idx
        metadata['Wavelengths'] = 'All' if wavelength_idx is None else wavelength_idx


        # read the file options
        file_list, sett_list = create_zemax_file_list(which_system=files_opt['which_system'],
                                                      AO_modes=files_opt['AO_modes'], scales=files_opt['scales'],
                                                      IFUs=files_opt['IFUs'], grating=files_opt['grating'])

        # Loop over the Zemax files
        results = []
        for zemax_file, settings in zip(file_list, sett_list):

            list_results = self.run_analysis(analysis_function=self.analysis_function_rms_wfe,
                                             files_dir=files_dir, zemax_file=zemax_file, results_path=results_path,
                                             results_shapes=results_shapes, results_names=results_names,
                                             wavelength_idx=wavelength_idx, configuration_idx=configuration_idx,
                                             surface=surface, spaxels_per_slice=spaxels_per_slice)

            results.append(list_results)
            rms_wfe, obj_xy, foc_xy, global_xy, wavelengths = list_results

            # Post-Processing the results
            file_name = zemax_file.split('.')[0]
            results_dir = os.path.join(results_path, file_name)
            settings['surface'] = 'IMG' if surface is None else surface

            self.save_hdf5(analysis_name='RMS_WFE', analysis_metadata=metadata, list_results=list_results, results_names=results_names,
                           file_name=file_name, file_settings=settings, results_dir=results_path)

            if plots is True:
                self.plot_and_save(analysis_name='RMS_WFE', list_results=list_results, file_name=file_name,
                                   file_settings=settings, results_dir=results_dir, wavelength_idx=wavelength_idx)





        return results




    def plot_RMS_WFE_maps(self, file_name, result_path, rms_ifu, wavelengths, colormap='jet'):

        N_waves = rms_ifu.shape[0]
        results_dir = os.path.join(result_path, file_name)

        for i in range(N_waves):
            wavelength = wavelengths[i]
            data = rms_ifu[i]
            minRMS = np.nanmin(data)
            maxRMS = np.nanmax(data)
            meanRMS = np.nanmean(data)

            plt.figure()
            plt.imshow(data, cmap=colormap, origin='lower')
            plt.title(r'$\lambda$: %.3f $\mu$m | RMS WFE [nm] | min:%.0f, max:%.0f, mean:%.0f nm' % (
            wavelength, minRMS, maxRMS, meanRMS))
            plt.clim(0, maxRMS)
            plt.colorbar(orientation='horizontal')
            plt.xlabel('Spaxels')
            plt.ylabel('Slices')
            fig_name = file_name + '_' + str(i + 1) + '_RMSWFE'
            if os.path.isfile(os.path.join(results_dir, fig_name)):
                os.remove(os.path.join(results_dir, fig_name))
            plt.savefig(os.path.join(results_dir, fig_name))
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

        # We would create way to many figures so we want to close them as we go

        return


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Create a Python Standalone Application
    psa = PythonStandaloneApplication()

    files_path = os.path.abspath("D:\End to End Model\April_2020")
    results_path = os.path.abspath("D:\End to End Model\Results_April")

    analysis = GeometricFWHM_PSF_Analysis(zosapi=psa)
    spaxel_scale = '60x30'
    focal_plane = focal_planes['IFS'][spaxel_scale]['PO']
    options = {'which_system': 'IFS', 'AO_modes': [], 'scales': ['60x30'], 'IFUs': ['EF'],
               'grating': ['VIS']}
    list_results = analysis.loop_over_files(files_dir=files_path, files_opt=options, results_path=results_path,
                                            wavelength_idx=[1], configuration_idx=None, surface=focal_plane,
                                            N_rays=35)

    fwhm, obj_xy, foc_xy, wavelengths = list_results

    plt.show()

    """ Old FWHM bits with Seaborn Kernel Density Estimation """

    # # For the FWHM we use Seaborn to calculate a Kernel Density Estimate
    # # Given the Raytrace positions (x, y) for a cloud of Rays, it calculates
    # # the density contours.
    # N_levels = 20
    #
    # # ax = axes[j]                # select the subplot axis to place the results
    # # sns.set_style("white")
    # # # contour = sns.kdeplot(x, y, n_levels=N_levels, ax=ax)       # Kernel Density Estimate with Seaborn
    # # contour = sns.kdeplot(x, y, n_levels=N_levels, ax=ax)       # Kernel Density Estimate with Seaborn
    # # half_contour = contour.collections[N_levels//2]             # Select the 50% level contour
    # # path = half_contour.get_paths()[0]                          # Select the path for that contour
    # # vertices = path.vertices                                    # Select vertices of that contour
    # # N_segments = vertices.shape[0]
    # #
    # # ax.scatter(x, y, color='blue', s=1)                         # Plot the cloud of Rays
    # # # ax.scatter(ref_x, ref_y, label=reference, color='green')  # Plot the reference point
    # # ax.scatter(vertices[:, 0], vertices[:, 1], color='red', s=5)
    # # if N_segments < 20:     # Warn me that the contour has too few vertices
    # #     print("Warning: Too few points (%d) along the 50 percent contour" % N_segments)
    # #     print("Results may be innacurate")
    # #
    # #     # new contour
    # #     next_contour = contour.collections[N_levels // 2 - 1]
    # #     next_path = next_contour.get_paths()[0]
    # #     next_vertices = next_path.vertices
    # #     N_next = next_vertices.shape[0]
    # #     print("Next Contour: %d" % N_next)
    #
    # # Old approach that sometimes fails because it uses very few point in the contour
    # # half_segments = half_contour.get_segments()[0]
    # # half_positions = half_contour.get_positions()
    #
    # # # Once we have the points that define the 50% contour we can use that to
    # # # calculate the FWHM
    # # radii = np.sqrt((vertices[:, 0] - ref_x) ** 2 + (vertices[:, 1] - ref_y) ** 2)
    # # mean_fwhm = 2 * 1e3 * np.mean(radii)            # Twice the mean distance between contour and ref. point
    # #
    # # # Marginal distributions - Across and Along the slice
    #
    # sns.set_style("white")
    # joint = sns.jointplot(x=x, y=y, kind='kde')
    #
    #
    # def calculate_marginal_fwhm(ax_marg, mode='X'):
    #     # We need the axis of the plot where
    #     # the marginal distribution lives
    #
    #     path_marg = ax_marg.collections[0].get_paths()[0]
    #     vert_marg = path_marg.vertices
    #     N_vert = vert_marg.shape[0]
    #     # the plot is a closed shape so half of it is just a path along y=0
    #     if mode == 'X':
    #         xdata = vert_marg[N_vert // 2:, 0]
    #         ydata = vert_marg[N_vert // 2:, 1]
    #     if mode == 'Y':  # In Y mode the data is rotated 90deg so X data becomes Y
    #         xdata = vert_marg[N_vert // 2:, 1]
    #         ydata = vert_marg[N_vert // 2:, 0]
    #
    #     peak = np.max(ydata)
    #     # as the sampling is constant we will count how many points are
    #     # above 50% of the peak, and how many below
    #     # and scale that ratio above/below by the range of the plot
    #     deltaX = np.max(xdata) - np.min(xdata)  # the range of our data
    #     above = np.argwhere(ydata >= 0.5 * peak).shape[0]
    #     below = np.argwhere(ydata < 0.5 * peak).shape[0]
    #     total = above + below
    #     # Above / Total = FWHM / Range
    #     fwhm = above / total * deltaX
    #
    #     return fwhm
    #
    #
    # fwhm_x = 1e3 * calculate_marginal_fwhm(joint.ax_marg_x, mode='X')
    # fwhm_y = 1e3 * calculate_marginal_fwhm(joint.ax_marg_y, mode='Y')
    #
    # # print("%.1f, %.1f, %.1f" % (mean_fwhm, fwhm_x, fwhm_y))
    #
    # # # And the projected sizes both ALONG (X) and ACROSS (Y) the slice
    # # rx = np.max(vertices[:, 0]) - np.min(vertices[:, 0])        # Max(x) - Min(x)
    # # ry = np.max(vertices[:, 1]) - np.min(vertices[:, 1])        # Max(y) - Min(y)
    # # fwhm_x, fwhm_y = 1e3 * np.mean(rx), 1e3 * np.mean(ry)
    #
    # # rx = np.sqrt((vertices[:, 0] - ref_x) ** 2)
    # # ry = np.sqrt((vertices[:, 1] - ref_y) ** 2)
    #
    # FWHM[j, :] = [-99, fwhm_x, fwhm_y]  # Store the results
    # # ax.set_title(r"Field %d | FWHM=%.1f, Fx$=%.1f, Fy=%.1f \mu$m" % (j+1, mean_fwhm, fwhm_x, fwhm_y))
    # # ax.set_aspect('equal')
    # # plt.legend()
    #
    # plt.close(joint.fig)

    del psa
    psa = None



