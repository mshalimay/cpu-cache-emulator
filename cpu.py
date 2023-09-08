from address import Address
from cache import Cache

class CPU():
    def __init__(self, cache:Cache, address:Address):
        self.address = address
        self.cache = cache
        self.instruction_count = 0

    def load_double(self, byte_address:int) -> float:
        """ Loads a double at the given RAM byte address into a register
        Args:
            byte_address (int): The byte address where the double is stored in RAM
        Returns:
            float: a double
        """
        self.address.set_address(byte_address)
        self.instruction_count += 1
        return self.cache.get_double(self.address)

    def store_double(self, byte_address:int, value:float) -> None:
        """ Writes the to the double at the given RAM byte
        Args:
            byte_address (int): The RAM byte address where value should be stored 

        Returns:
            float: a double
        """
        self.address.set_address(byte_address)
        self.cache.set_double(self.address, value)
        self.instruction_count += 1
        

    def add_double(self, value1:float, value2:float) -> float:
        """Add two doubles and return the result"""
        self.instruction_count += 1
        return value1 + value2


    def mult_double(self, value1:float, value2:float) -> float:
        """Multiply two doubles and return the result"""
        self.instruction_count += 1
        return value1 * value2
        
