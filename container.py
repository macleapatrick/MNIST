import torch

class ImageFile():
    def __init__(self, file):
        self.file = file
        self.data = []

    def read(self):
        """
        reads idx imagefile
        """
        with open(self.file, "rb") as file:
            magic = self.bytes_to_int(file.read(4))
            length = self.bytes_to_int(file.read(4))
            rows = self.bytes_to_int(file.read(4))
            cols = self.bytes_to_int(file.read(4))
            size = rows*cols

            for _ in range(length):
                self.data.append([x for x in file.read(size)])

    def bytes_to_int(self, bytes):
        return int.from_bytes(bytes, byteorder="big")

    def get_array(self):
        return self.data
    
class LabelFile():
    def __init__(self, file):
        self.file = file
        self.data = []

    def read(self):
        """
        reads idx label file
        """
        with open(self.file, "rb") as file:
            magic = self.bytes_to_int(file.read(4))
            length = self.bytes_to_int(file.read(4))

            for _ in range(length):
                label = self.bytes_to_int(file.read(1))
                self.data.append([x == label for x in range(10)])

    def bytes_to_int(self, bytes):
        return int.from_bytes(bytes, byteorder="big")

    def get_array(self):
        return self.data
                




