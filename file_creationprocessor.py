import json

class FileCreationProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def create_file(self, data):
        with open(self.file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"File created at: {self.file_path}")