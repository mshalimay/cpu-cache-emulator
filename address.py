
# NOTE:this class represents a WORD address 
# it receives a byte address and calculates the word address
# this is only to simplify because the RAM is "double" addressable
# could also hold the byte address and do the conversion before bitmasking to retrieve the tag, index and block offset
# conversion = move the bit representation of the byte address 'log_2(word_size)' bits to the right or int(byte_address/word_size)

class Address:
    def __init__(self, byte_address:int, bits_for_word_address:int, bits_for_block_offset, bits_for_index:int, word_size_bytes:int):
        self.address = byte_address // word_size_bytes
        self.word_size = word_size_bytes

        # define the bit positions for the block offset, index and tag in the WORD address
        # eg: byte address =  1010 100 11 101, word_size = 8 bytes
        # int(byte_address/8) => shift bin representation three for the right
        # ==> word address = 1010 100 11 (bits used for address = 8)

        # bit_offset_0 = 0 , bit_offset_1 = 1 --> '11' from 101010011
        self.bit_offset_0 = 0
        self.bit_offset_1 = bits_for_block_offset - 1

        # bit_index_0 = 2, bit_index_1 = 4 -> '100' from 101010011
        self.bit_index_0 = bits_for_block_offset
        self.bit_index_1 = bits_for_block_offset + bits_for_index - 1

        # bit_tag_0 = 5, bit_tag_1 = 8 --> '1010' from 101010011
        self.bit_tag_0 = bits_for_block_offset + bits_for_index
        self.bit_tag_1 = bits_for_word_address - 1

    def set_address(self, byte_address:int):
        """ Receives a byte address, calculate accordingly word address and sets it"""
        self.address = byte_address // self.word_size

    def get_tag(self) -> int:
        """Computes the tag bits from the word address using bitmasking"""
        tag_mask = (1 << (self.bit_tag_1 - self.bit_tag_0 + 1)) - 1
        return (self.address >> (self.bit_index_1 + 1)) & tag_mask

    def get_index(self) -> int:
        """Computes the index bits from the word address using bitmasking"""
        index_mask = (1 << (self.bit_index_1 - self.bit_index_0 + 1)) - 1
        return (self.address >> self.bit_offset_1 + 1) & index_mask
        
    def get_block_offset(self) -> int:
        """Computes the index bits from the word address using bitmasking"""
        block_offset_mask = (1 << (self.bit_offset_1 - self.bit_offset_0 + 1)) - 1
        return self.address & block_offset_mask
    