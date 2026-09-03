import pandas as pd
class Participant:
    def __init__(self, file_input):
        self.name:str = ""
        self.raw_bytes:bytes = self.process_file(file_input)

        # This keeps track of whether data have been parsed
        self.is_processed = False
        self.data_frame: pd.DataFrame = pd.DataFrame()

    def get_name(self) -> str:
        if not self.name:
            print("Failed, Fetch name before getting data.")
            raise ValueError("Method failed, Please load the participant's data before getting the name.")
        return self.name

    def get_raw_bytes(self) -> bytes:
        if not self.raw_bytes:
            print("Failed no file uploaded")
            raise ValueError("Method failed, Please load the participant's data before getting the raw_bytes.")
        return self.raw_bytes

    def get_data_frame_items(self) -> list:
        output_data = [ i.to_dict() for i in self.data_frame ]
        return output_data

    def is_processed(self):
        return self.is_processed

    def load_data_frame(self, df: pd.DataFrame) -> None:
        """TODO: When name already exist ask if name want to be replace"""
        if not df.empty:
            self.data_frame = df
            self.is_processed = True
            self.raw_bytes = None

            if self.name:
                pass
            self.set_name(df["Participant"].iloc[0])

    def set_name(self, name):
        if not name:
            raise ValueError("Method failed, Name can not be null")
        self.name = name

    def process_file(self, file_input):
        """
        Processes file data from either a local file path, a Path object,
        or an in-memory Streamlit UploadedFile buffer.
        """
        if isinstance(file_input, str):
            # Handle physical file path on disk
            with open(file_input, "rb") as f:
                data_bytes = f.read()

        elif hasattr(file_input, "read"):
            # Handle Streamlit UploadedFile or io.BytesIO buffer
            data_bytes = file_input.read()

            # CRITICAL: Reset pointer if you plan to read the buffer again later
            if hasattr(file_input, "seek"):
                file_input.seek(0)
        else:
            raise TypeError("Unsupported file input type. Must be str, Path, or file-like object.")

        return data_bytes
