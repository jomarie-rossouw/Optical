import os
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def save_fit(output_root, x, y, x_name, y_name, res, fitted_eq, param_values, extra_metadata=None):
    output_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_dir / 'data.csv',
               np.column_stack((x, y)),
               delimiter=',',
               header=f'{x_name}, {y_name}',
               comments='')

    with open(output_dir / 'fit_report.txt', 'w') as f:
        f.write(res.fit_report())

    with open(output_dir / 'fitted_equation.txt', 'w') as f:
        f.write(str(fitted_eq))

    initial_params = {
        k: float(v) if np.isscalar(v) else np.asarray(v).tolist()
        for k, v in param_values.items()
        if k != x
    }
    with open(output_dir / 'initial_parameters.json', 'w') as f:
        json.dump(initial_params, f, indent=2)

    with open(output_dir / 'best_parameters.json', 'w') as f:
        json.dump(res.params.valuesdict(), f, indent=2)

    repro_info = {
        'script': str(Path(__file__).resolve()),
        'date': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'matplotlib_version': plt.__version__,
    }
    if extra_metadata:
        repro_info.update(extra_metadata)

    with open(output_dir / 'reproduction_info.json', 'w') as f:
        json.dump(repro_info, f, indent=2)

    fig = plt.figure()
    res.plot_fit(fig=fig)
    fig.savefig(output_dir / 'fit_plot.png', dpi=300)
    plt.close(fig)

    return output_dir