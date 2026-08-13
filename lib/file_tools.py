import json
import csv
from participant import Participant
import pandas as pd

def load_participant(path):
    """Load a single participant file"""
    """TODO: Differentiate between model and where data want to be loaded"""
    converter = ExcelConverter(path)
    model = Participant()
    # Run the pipeline
    converter.load_raw_data_and_map(model)
    print(f"{model.get_name()} Conversion complete!")
    return model

def load_participants_file_all(path):
    """Load all participant file under the path"""
    """TODO: Finish Writing and find out how to find all paths"""
    converter = ExcelConverter(path)

    # Run the pipeline


class ExcelConverter:
    """Handles the file reading, object mapping, and file saving."""

    def __init__(self, file_path):
        self.file_path = self.validate_file_path(file_path)
        self.objects_list = []

    def validate_file_path(self, path):
        """TODO: Validate Paths"""
        return path

    def load_raw_data_and_map(self, model):
        """Reads CSV and maps rows to RowData objects."""
        """TODO: After load_participant done fix"""
        data = pd.read_csv(self.file_path)
        model.load_data(data)
