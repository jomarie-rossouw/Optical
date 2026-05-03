import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import re
import matplotlib.pyplot as plt
from natsort import natsorted
from lmfit import Parameters, Minimizer, minimize, report_fit
from lmfit.models import LorentzianModel, GaussianModel, ConstantModel, ExpressionModel
from lmfit.model import save_modelresult, save_model, load_model, load_modelresult
from tabulate import tabulate
import math

def load_tdpl_data(path, filename_pattern=r"\_S1/(.*?)\ ", temp_offset=273):
    """
    Load TDPL data from CSV files in the given path.

    Parameters:
    - path: str, path to the directory containing CSV files
    - filename_pattern: str, regex pattern to extract temperature from filename
    - temp_offset: int, offset to add to extracted temperature (e.g., 273 for K)

    Returns:
    - all_dfs: pd.DataFrame, concatenated data with energy as index and temp as columns
    - temp: list, sorted list of temperatures
    - nm: np.array, wavelength data
    - eV: np.array, energy data
    """
    fns = glob.glob(path + '*.csv')
    if not fns:
        raise ValueError("No CSV files found in the specified path.")

    temp = []
    for file in fns:
        match = re.search(filename_pattern, file)
        if match:
            kelvin = int(match.group(1)) + temp_offset
            temp.append(kelvin)
        else:
            raise ValueError(f"Could not extract temperature from filename: {file}")

    # Load wavelength and energy
    nm = np.loadtxt(fns[0], delimiter=',', usecols=0, skiprows=2)
    eV = 1240 / nm

    # Load data
    all_dfs = pd.concat([pd.read_csv(one_filename, usecols=[2], skiprows=[0])
                         for one_filename in fns], axis=1).set_index(eV)
    all_dfs.columns = temp
    all_dfs = all_dfs.reindex(natsorted(all_dfs.columns), axis=1)
    temp = natsorted(temp)

    return all_dfs, temp, nm, eV

def process_data(all_dfs, energy_min, energy_max):
    """
    Section the data based on energy range.
    Returns:
    - df_sectioned: pd.DataFrame, sectioned data
    """
    df_sectioned = all_dfs[all_dfs.index.to_series().between(energy_min, energy_max)]
    return df_sectioned
### peak fitting functions
#-----------------------------------------------------------------------------------------------------------------------------
def fit_peaks(x, y, model_components, add_constant=True):
    """
    Fit peaks to the data using lmfit with customizable model components.

    Parameters:
    - x: np.array, energy values
    - y: np.array, intensity values
    - model_components: list of dicts, each specifying a model component
      Each dict should have:
      - 'type': str, 'gaussian', 'lorentzian', or 'expression'
      - 'prefix': str, unique prefix for parameters
      - 'label': str, plot label for the component
      - 'color': str, matplotlib color for the component
      - 'params': dict, parameter values and bounds (for gaussian/lorentzian)
      - 'expr': str, expression for ExpressionModel
    - add_constant: bool, whether to add a constant offset

    Returns:
    - result: lmfit ModelResult
    - plot_info: list of dicts with name, label, and color for each component
    """
    model = None
    params = Parameters()
    plot_info = []

    if add_constant:
        const_model = ConstantModel()
        model = const_model
        params.update(const_model.make_params())
        plot_info.append({'name': 'c', 'label': 'Constant', 'color': 'black'})

    for comp in model_components:
        comp_type = comp['type']
        prefix = comp.get('prefix', '')
        label = comp.get('label', prefix.rstrip('_'))
        color = comp.get('color', None)

        if comp_type == 'gaussian':
            m = GaussianModel(prefix=prefix)
        elif comp_type == 'lorentzian':
            m = LorentzianModel(prefix=prefix)
        elif comp_type == 'expression':
            expr = comp.get('expr', '')
            m = ExpressionModel(expr)
        else:
            raise ValueError(f"Unsupported model type: {comp_type}")

        p = m.make_params()
        comp_params = comp.get('params', {})
        for key, val in comp_params.items():
            if key in p:
                p[key].set(**val if isinstance(val, dict) else {'value': val})

        if model is None:
            model = m
        else:
            model += m

        params.update(p)
        plot_info.append({'name': m.name, 'label': label, 'color': color})

    if model is None:
        raise ValueError("No model components specified")

    def objective(params, x, data):
        resid = data - model.eval(params=params, x=x)
        return resid.flatten()

    out = minimize(objective, params=params, args=(x, y))
    result = model.fit(data=y, params=out.params, x=x)

    return result, plot_info

def calculate_peak_centroids(x, result, component_names=None):
    """
    Calculate the centroid energy for fitted peak components.

    Parameters:
    - x: np.array, energy values
    - result: lmfit ModelResult
    - component_names: list of component names to calculate centroids for.
      If None, all fitted components are used.

    Returns:
    - centroids: dict mapping component name to centroid energy
    """
    comps = result.eval_components()
    if component_names is None:
        component_names = list(comps.keys())

    centroids = {}
    for name in component_names:
        if name not in comps:
            raise ValueError(f"Component '{name}' not found in fit result components")
        comp = comps[name]
        total = np.sum(comp)
        if total == 0:
            centroids[name] = np.nan
        else:
            centroids[name] = float(np.sum(x * comp) / total)
    return centroids

def plot_and_save(x, y, result, temp, save_prefix='S1', show_plot=False, plot_info=None, output_dir=None, xlim=None, ylim=None):
    """
    Plot the fit results and save to file.

    Parameters:
    - x: np.array, energy
    - y: np.array, data
    - result: lmfit ModelResult
    - temp: int, temperature
    - save_prefix: str, prefix for save files
    - show_plot: bool, whether to show the plot
    - plot_info: list of dicts with 'name', 'label', 'color'
    - output_dir: str, directory to save files
    - xlim: tuple or list, x-axis limits
    - ylim: tuple or list, y-axis limits
    """
    if output_dir is None:
        output_dir = os.getcwd()

    comps = result.eval_components()
    plt.figure()
    plt.plot(x, y, '-', color='gray', label='Data')
    plt.plot(x, result.best_fit, '--', label='Fit', color='black')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if plot_info is not None:
        comps_items = list(comps.items())
        if len(plot_info) == len(comps_items):
            for (name, comp), info in zip(comps_items, plot_info):
                label = info.get('label', name)
                color = info.get('color', None)
                plt.plot(x, comp, '--', label=label, color=color)
        else:
            color_map = {info['name']: {'label': info.get('label', info['name']), 'color': info.get('color', None)} for info in plot_info}
            for name, comp in comps_items:
                entry = color_map.get(name, {})
                label = entry.get('label', name)
                color = entry.get('color', None)
                plt.plot(x, comp, '--', label=label, color=color)
    else:
        for name, comp in comps.items():
            plt.plot(x, comp, '--', label=name)

    plt.xlabel("Energy (eV)")
    plt.ylabel('PL Intensity (a.u.)')
    plt.title(f'{temp} K')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{save_prefix}_{temp}K.png'))
    if show_plot:
        plt.show()
    plt.close()

    # Save report
    results = [['Parameter', 'Value']]
    with open(os.path.join(output_dir, f'{save_prefix}_{temp}_report.txt'), 'w') as f:
        f.write(f'{temp} K.\n')
        f.write('---------------------------------\n')
        for name, param in result.params.items():
            results.append([f'{name}', f'{param.value:.5f}'])
        f.write(tabulate(results, headers='firstrow'))
        f.write('\n---------------------------------\n')

def generate_comprehensive_report(result, model_components, temp, filename, x=None):
    """
    Generate a comprehensive report file containing model components, fit report, centroids, and equations.

    Parameters:
    - result: lmfit ModelResult
    - model_components: list, input model components
    - temp: int, temperature
    - filename: str, output filename
    - x: np.array, energy values for centroid calculation
    """
    with open(filename, 'w') as f:
        f.write(f'Comprehensive Fit Report for {temp} K\n')
        f.write('=' * 50 + '\n\n')

        # Model Components
        f.write('Model Components:\n')
        f.write('-' * 17 + '\n')
        for i, comp in enumerate(model_components):
            f.write(f'Component {i+1}: {comp}\n')
        f.write('\n')

        # Fit report
        f.write('Fit Report:\n')
        f.write('-' * 12 + '\n')
        f.write(result.fit_report())
        f.write('\n')

        # Centroids
        f.write('Peak Centroids (emission energies):\n')
        f.write('-' * 30 + '\n')
        if x is not None:
            centroids = calculate_peak_centroids(x, result)
            for name, centroid in centroids.items():
                f.write(f'{name}: {centroid:.6f} eV\n')
        else:
            f.write('Energy axis not provided; centroid calculation skipped.\n')
        f.write('\n')

        # Equations
        f.write('Fitted Equations:\n')
        f.write('-' * 17 + '\n')
        params = result.params

        for comp in model_components:
            prefix = comp.get('prefix', '')
            comp_type = comp['type']
            if comp_type == 'gaussian':
                amp_key = f'{prefix}amplitude'
                cen_key = f'{prefix}center'
                sig_key = f'{prefix}sigma'
                if amp_key in params and cen_key in params and sig_key in params:
                    amp = params[amp_key].value
                    cen = params[cen_key].value
                    sig = params[sig_key].value
                    f.write(f'Gaussian ({prefix[:-1]}): f(x) = {amp:.5f} * exp(- (x - {cen:.5f})^2 / (2 * {sig:.5f}^2))\n')
            elif comp_type == 'lorentzian':
                amp_key = f'{prefix}amplitude'
                cen_key = f'{prefix}center'
                sig_key = f'{prefix}sigma'
                if amp_key in params and cen_key in params and sig_key in params:
                    amp = params[amp_key].value
                    cen = params[cen_key].value
                    sig = params[sig_key].value
                    f.write(f'Lorentzian ({prefix[:-1]}): f(x) = {amp:.5f} * ({sig:.5f}^2 / ((x - {cen:.5f})^2 + {sig:.5f}^2)) / ({math.pi:.5f} * {sig:.5f})\n')
            elif comp_type == 'expression':
                expr = comp.get('expr', '')
                f.write(f'Expression ({prefix[:-1]}): f(x) = {expr}\n')
                # Note: parameters are fitted, but expression is symbolic

        # Constant
        if 'c' in params:
            c = params['c'].value
            f.write(f'Constant: f(x) = {c:.5f}\n')

        f.write('\nTotal Model: ' + ' + '.join([f"{comp['type']} ({comp.get('prefix', '')[:-1]})" for comp in model_components]) + (' + Constant' if 'c' in params else '') + '\n')

def analyze_tdpl(path, energy_min=2.20, energy_max=2.55, model_components=None, save_prefix='S1', show_plot=False, add_constant=True, output_dir=None, xlim=None, ylim=None):
    """
    Main function to analyze TDPL data.

    Parameters:
    - path: str, path to data
    - energy_min: float
    - energy_max: float
    - model_components: list of dicts, model components (see fit_peaks)
    - save_prefix: str
    - show_plot: bool
    - add_constant: bool, add constant offset
    - output_dir: str, directory to save outputs (default: current dir)
    - xlim: tuple or list, x-axis limits for plots
    - ylim: tuple or list, y-axis limits for plots
    """
    import os
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)

    if model_components is None:
        # Default model: Gaussian + 2 Lorentzians
        model_components = [
            {'type': 'gaussian', 'prefix': 'g1_', 'params': {'center': {'value': 2.33, 'min': energy_min, 'max': energy_max}, 'amplitude': {'value': 1.11, 'min': 0}}},
            {'type': 'lorentzian', 'prefix': 'l1_', 'params': {'center': {'value': 2.364, 'min': energy_min, 'max': energy_max}, 'amplitude': {'value': 1.657, 'min': 0}}},
            {'type': 'lorentzian', 'prefix': 'l2_', 'params': {'center': {'value': 2.38, 'min': energy_min, 'max': energy_max}, 'amplitude': {'value': 1.2, 'min': 0}}}
        ]

    all_dfs, temp, nm, eV = load_tdpl_data(path)
    df_sectioned = process_data(all_dfs, energy_min, energy_max)
    data = df_sectioned.to_numpy()

    for i in range(len(temp)):
        y = data[:, i]
        x = df_sectioned.index.to_series().to_numpy()

        result, plot_info = fit_peaks(x, y, model_components, add_constant)
        report_fit(result)
        plot_and_save(x, y, result, temp[i], save_prefix, show_plot, plot_info, output_dir, xlim=xlim, ylim=ylim)
        generate_comprehensive_report(result, model_components, temp[i], os.path.join(output_dir, f'{save_prefix}_{temp[i]}_comprehensive_report.txt'), x)

def analyze_multiple_datasets(dataset_paths, energy_min=2.20, energy_max=2.55, model_components=None, save_prefix='S1', show_plot=False, add_constant=True, base_output_dir=None, xlim=None, ylim=None):
    """
    Run TDPL analysis for multiple dataset directories.

    Parameters:
    - dataset_paths: list of str
    - energy_min: float
    - energy_max: float
    - model_components: list of dicts
    - save_prefix: str
    - show_plot: bool
    - add_constant: bool
    - base_output_dir: str, directory where all dataset output subfolders will be created
    - xlim: tuple or list, x-axis limits for plots
    - ylim: tuple or list, y-axis limits for plots

    Returns:
    - list of output directories created
    """
    output_dirs = []
    for path in dataset_paths:
        dataset_name = os.path.basename(os.path.normpath(path)) or 'dataset'
        if base_output_dir is None:
            dataset_output_dir = os.path.join(os.getcwd(), f'{dataset_name}_{save_prefix}')
        else:
            dataset_output_dir = os.path.join(base_output_dir, f'{dataset_name}_{save_prefix}')
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir, exist_ok=True)

        analyze_tdpl(
            path=path,
            energy_min=energy_min,
            energy_max=energy_max,
            model_components=model_components,
            save_prefix=f'{dataset_name}_{save_prefix}',
            show_plot=show_plot,
            add_constant=add_constant,
            output_dir=dataset_output_dir,
            xlim=xlim,
            ylim=ylim
        )
        output_dirs.append(dataset_output_dir)

    return output_dirs

#------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    path = "/home/jo-marie/Documents/faafo/PL/TDPL/Glass1_S1/"
    analyze_tdpl(path, xlim=(2.20, 2.55), output_dir="/home/jo-marie/Documents/faafo/PL/TDPL/Glass1_S1/fit_results/")

