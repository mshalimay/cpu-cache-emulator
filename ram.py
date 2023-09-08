from data_block import DataBlock

# RAM is a flat sequence of DataBlocks
class RAM:
    # data:list[DataBlock]
    def __init__(self, num_blocks:int, data):
        self.num_blocks = num_blocks
        self.data = data
        self.addressable_units_per_block = len(data[0].data)

    def get_block(self, address:int) -> DataBlock:
        """ Gets the data block at the given word address
        Args:
            address (int): The word address to get the data block from.
            Notice: not the byte address; word address = byte address / size of double in bytes

        Returns:
            DataBlock: The data block at the given word address
        """
        ram_data_block_num = address // self.addressable_units_per_block
        return self.data[ram_data_block_num]

    def set_block(self, address:int, data_block:DataBlock):
        """ Sets the data block at the given word address

        Args:
            address (int): The word address to set the data block at.
            Notice: not the byte address; word address = byte address / size of double in bytes
            data_block (DataBlock): The data block to set at the given word address
        """
        ram_data_block_num = address // self.addressable_units_per_block
        self.data[ram_data_block_num] = data_block
    
        