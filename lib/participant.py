class Participant:
    def __init__(self):
        self.name:str = ""
        self.data:list[ParticipantRow] = []

    def get_name(self) -> str:
        if not self.name:
            print("Failed, Fetch name before getting data.")
            raise ValueError("Method failed, Please load the participant's data before getting the name.")
        return self.name

    def set_name(self, name):
        if not name:
            print("Failed, No name given")
            raise ValueError("Method failed, Name can not be null")
        self.name = name

    def get_data_items(self) -> list:
        output_data = [ i.to_dict() for i in self.data ]
        return output_data

    def load_data(self, data) -> None:
        """TODO: When name already exist ask if name want to be replace"""
        if not data.empty:
            self.data = data
            if self.name:
                pass

            self.set_name(data["Participant"].iloc[0])

class ParticipantRow:
    def __init__(self, data_dict:dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        return self.__dict__

    def get(self, k:str) -> any:
        v = None
        if k:
            v = self.__getattribute__(k)
        return v