from lib.participant import Participant
from pathlib import Path
import pandas as pd
import io

# Designed to serve Table Related Function
def load_participants(participant_objs: list) -> None:
    """Load participants binary file"""
    converter = ExcelConverter()
    for p_obj in participant_objs:
        if not p_obj.is_processed:
            data_frame = converter.load_raw_data(p_obj.get_raw_bytes())
            p_obj.load_data_frame(data_frame)
    print(f"All participants loaded")


def create_participants(uploaded_files: list) -> list:
    """
    Processes all given files (paths or uploads).
    Creates and returns as many successful Participant objects as possible.
    """
    successful_participants = []
    converter = ExcelConverter()

    for file in uploaded_files:
        try:
            name = converter.peak_first_field(file, 'Participant')
            model = Participant(file)
            model.set_name(name)
            successful_participants.append(model)

        except Exception as e:
            # Catch errors for this specific file so the loop can continue
            print(f"Skipping file {getattr(file, 'name', file)} due to error: {e}")
            continue

    print(f"Participants created")
    return successful_participants


# General Tools
class ExcelConverter:
    """Handles the file reading, object mapping, and file saving."""

    def __init__(self):
        self.objects_list = []

    def validate_uploaded_file(self, file_input):

        if hasattr(file_input, "seek"):
            file_input.seek(0)

        if isinstance(file_input, bytes):
            file_like_object = io.BytesIO(file_input)

            return file_like_object

        if isinstance(file_input, (str, Path)):
            filename = str(file_input)
        else:
            filename = getattr(file_input, 'name', None) or getattr(file_input, 'filename', '')

        if not filename or Path(filename).suffix.lower() != '.csv':
            raise ValueError("Invalid file type. Only CSV files are allowed.")

        return file_input

    def peak_first_row(self, uploaded_file, field):

        file = self.validate_uploaded_file(uploaded_file)

        df = pd.read_csv(file, nrows=1, header=0)

        if hasattr(file, "seek"):
            file.seek(0)

        if df.empty:
            return None

        if field:
            col_idx = df.columns.get_loc(field)
            return df.iloc[0, col_idx]

        return df.iloc[0]

    def peak_first_field(self, uploaded_file, field):

        if not field:
            raise ValueError("Field arg is needed, use peak_first_col for full row")

        first_col_series = self.peak_first_row(uploaded_file, field)

        if not first_col_series:
            return None

        return first_col_series

    def load_raw_data(self, uploaded_file) -> pd.DataFrame:
        file = self.validate_uploaded_file(uploaded_file)
        df = pd.read_csv(file)

        # Clean up after reading the csv
        if hasattr(file, "seek"):
            file.seek(0)

        return df
