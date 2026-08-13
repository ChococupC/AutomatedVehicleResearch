import json
from lib.file_tools import load_participant
from lib.participant import Participant
import matplotlib.pyplot as plt
from lib.graph import process_toc_experiment_data, plot_manual_percentage_by_scenario, plot_prior_failure_ttc_impact
from pathlib import Path
import pandas as pd
import random

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Replace with your folder path
    folder = Path("data")

    # Generates a list of full paths for files only
    file_paths = [str(file.resolve()) for file in folder.iterdir() if file.is_file()]
    datas = []
    sample_paths = random.sample(file_paths, 30)
    for path in sample_paths:
        datas.append(load_participant(path).data)
    combined_df = pd.concat(datas, ignore_index=True)
    summary_df = process_toc_experiment_data(combined_df)
    summary_df.to_csv("output.csv", index=False)
    print(summary_df)
    graph = plot_prior_failure_ttc_impact(summary_df)
    plt.show()
