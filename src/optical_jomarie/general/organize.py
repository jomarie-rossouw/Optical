import os
from datetime import datetime

def save_comments(out_dir, data_name, data_path):
    # Create timestamped output directory
    output_dir = out_dir or os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"analysis_results/{dataset_name}_{timestamp}"
    

    with open(os.path.join(output_dir, f'{save_prefix}_{temp}_report.txt'), 'w') as f:
        f.write(f'{temp} K.\n')
        f.write('-' * 33 + '\n')
        f.write(tabulate(results, headers='firstrow'))
        f.write('\n' + '-' * 33 + '\n')

    print(f"Output directory: {output_dir}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    return output_dir