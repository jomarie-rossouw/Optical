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

if __name__ == '__main__':
    plot_scatter()