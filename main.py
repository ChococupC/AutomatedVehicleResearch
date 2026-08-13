import json
from lib.file_tools import load_participant
from lib.participant import Participant
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = "data/Sample_Scenario_Data_2025_04_24-04_39_PM.csv"
    participant = Participant()
    load_participant(data_path)
    participant.data.plot(x='Scenario_Name', y='Y_Pos',title="coordinate")
    plt.show()

