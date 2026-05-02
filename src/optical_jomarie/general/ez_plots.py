import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(data, xlabel, ylabel, title):
    x_data = data[:,0]
    y_data = data[:,1]

    max_value = max(y_data)
    max_position = x_data[np.where(y_data == max_value)]
    
    print(f'The maximum value is {max_value} and it occurs at {max_position}')
    plt.scatter(x_data, y_data, linewidths=0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_overlay(sample1, sample2, unit = 'ev', fov = 'close', ylimit = None):
    valid_energy = {'energy', 'Energy', 'ev', 'eV', 'E'}
    valid_lambda = {'wavelength', 'lambda', 'nm'}

    data1 = np.loadtxt(sample1, delimiter=',')
    nm = data1[:,0]
    ev = 1240/nm
    S1 = data1[:,2]

    data2 = np.loadtxt(sample2, delimiter=',')
    S2 = data2[:,2]

    if fov == 'close':
        x_nm = (400, 650)
        x_ev = (2, 3)
    elif fov == 'full':
        x_nm = (319, 1089)
        x_ev = (1.14, 3.8)
    else:
        raise ValueError("Invalid")

    if (unit in valid_energy) or (unit in valid_lambda):
        if unit in valid_energy:
            plt.plot(ev, S1, color = '#1f77b4', label = 'S1')
            plt.plot(ev, S2, color = '#ff7f0e', label = 'S2')
            plt.xlabel('Energy (eV)')
            plt.ylabel('Intensity (a.u.)')
            plt.xlim(x_ev)
            plt.ylim(ylimit)
            plt.legend()
            plt.show()
        if unit in valid_lambda:
            plt.plot(nm, S1, color = '#1f77b4', label = 'S1')
            plt.plot(nm, S2, color = '#ff7f0e', label = 'S2')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity (a.u.)')
            plt.xlim(x_nm)
            plt.ylim(ylimit)
            plt.legend()
            plt.show()
    else:
        raise ValueError('Invalid')


def plot_single(data, sample = 'S1', unit = 'ev', fov = 'close', ylimit = None):
    valid_energy = {'energy', 'Energy', 'ev', 'eV', 'E'}
    valid_lambda = {'wavelength', 'lambda', 'nm'}

    data1 = np.loadtxt(data, delimiter=',')
    nm = data1[:,0]
    ev = 1240/nm
    S = data1[:,2]

    if fov == 'close':
        x_nm = (400, 650)
        x_ev = (2, 3)
    elif fov == 'full':
        x_nm = (319, 1089)
        x_ev = (1.14, 3.8)
    else:
        raise ValueError("Invalid")

    if sample == 'S1':
        colour = '#1f77b4'
    elif sample == 'S2':
        colour = '#ff7f0e'
    else:
        raise ValueError('Sample value should either be ''S1'' or ''S2''')

      
    if (unit in valid_energy) or (unit in valid_lambda):
        if unit in valid_energy:
            plt.plot(ev, S, color = colour)
            plt.xlabel('Energy (eV)')
            plt.ylabel('Intensity (a.u.)')
            plt.xlim(x_ev)
            plt.ylim(ylimit)
            plt.show()
        elif unit in valid_lambda: 
            plt.plot(nm, S, color = colour)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity (a.u.)')
            plt.xlim(x_nm)
            plt.ylim(ylimit)
            plt.show()
    else:
        raise ValueError('Not a valid value for unit')


if __name__ == '__main__':
    data1 = '/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Glass1_S1/-190 -- 2026-Feb-16 12-17-08.csv'
    data2 = '/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Glass2_S1/-190 -- 2026-Feb-16 12-57-39.csv'
    plot_overlay(data1, data2, unit = 'nm')