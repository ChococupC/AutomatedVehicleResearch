import json
from lib.file_tools import create_participants, load_participants, ExcelConverter
from lib.participant import Participant
import matplotlib.pyplot as plt
from lib.graph import process_toc_experiment_data, filter_experiment_data
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

    converter = ExcelConverter()
    p_objs = create_participants([sample_paths[0]])
    load_participants(p_objs)
    df = p_objs[0].data_frame
    df.to_csv("output.csv", index=False)


    #
    # for i in you:
    #     print(i.get_name())
    # # for path in file_paths:
    # #     datas.append(load_participant(path).data)
    # # combined_df = pd.concat(datas, ignore_index=True)
    # # summary_df = filter_experiment_data(combined_df, Tailgate_Onset_min=2621, Timestep_min=2620, Warning=True, TTC_min=5)
    # # summary_df.to_csv("output.csv", index=False)
    # # print(summary_df)
    # # plt.show()
