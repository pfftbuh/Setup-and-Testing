import json

class FileCreationProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def create_file(self, data):
        with open(self.file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"File created at: {self.file_path}")
    
    def exists(self):
        try:
            with open(self.file_path, 'r') as file:
                return True
        except FileNotFoundError:
            return False
    
    def append_to_file(self, data):
        existing_data = []
        if self.exists():
            with open(self.file_path, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []
        existing_data.append(data)
        self.create_file(existing_data)