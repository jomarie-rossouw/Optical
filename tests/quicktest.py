import numpy as np
from optical_jomarie.general.ez_plots import plot_scatter

# synthetic data: Nx2 array
data = np.column_stack((np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50))))
plot_scatter(data, xlabel="x", ylabel="y", title="Test scatter")
