

# A data block of doubles
class DataBlock:
    # data:list[float]
    def __init__(self, size:int, data):
        self.size = size
        self.data = data
        self.tag = None
        self.valid = None

    def get_double(self, offset:int) -> float:
        """ Returns the double at the given offset in the data block
        Args:
            offset (int): The offset bits to access the double within the data block

        Returns:
            float: the value of the double at the given offset
        """
        return self.data[offset]

    def set_double(self, offset:int, value:float) -> None:
        """ Sets the double at the given offset in the data block to the given value

        Args:
            offset (int): The offset bits to access the double within the data block
            value (float): The value to set the double to
        """
        self.data[offset] = value