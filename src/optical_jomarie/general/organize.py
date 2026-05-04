import os
from datetime import datetime

def save_reults(data_name, data_path):
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"analysis_results/{dataset_name}_{timestamp}"
    
    print(f"Output directory: {output_dir}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    return output_dir