import numpy as np
import os

os.chdir("/home/mshalimay/AdvancedComputerArchitechture/Project1/")

from data_block import DataBlock
from cache import Cache
from ram import RAM
from address import Address
from cpu import CPU

WORD_SIZE_BYTES = 8
ram_size_bytes = 1024
cache_size_bytes = 256
data_block_size_bytes = 16
n_way_associativity = 2

# calculate DataBlock parameters
words_per_block = int(data_block_size_bytes / WORD_SIZE_BYTES)

# calculate RAM parameters and instantiate it
num_blocks = int(ram_size_bytes / data_block_size_bytes)
data_blocks = [DataBlock(size=data_block_size_bytes, data=[0.0] * words_per_block) for _ in range(num_blocks)]
ram = RAM(num_blocks=num_blocks, data = data_blocks)

# instantiate cache
num_cache_blocks = int(cache_size_bytes / data_block_size_bytes)
cache = Cache(n_way_associativity = n_way_associativity, num_blocks = num_cache_blocks, ram = ram, repl_policy = "LRU")

# calculate address parameters and instantiate it
max_word_address = int(ram_size_bytes / WORD_SIZE_BYTES - 1)
bits_for_word_address = int(np.log2(ram_size_bytes / WORD_SIZE_BYTES))
bits_for_block_offset = int(np.log2(words_per_block))
bits_for_index = int(np.log2(cache.num_sets))

address =  Address(byte_address=0, bits_for_word_address = bits_for_word_address, 
                   bits_for_block_offset=bits_for_block_offset, bits_for_index=bits_for_index, 
                   word_size=WORD_SIZE_BYTES)

# instantiate the CPU
cpu = CPU(cache = cache, address = address)

cpu.store_double(1023, 3.14)

x = cpu.load_double(1023)


#-------------------------------------------------------------------------------
# DEBUGGING
#-------------------------------------------------------------------------------

# testing the parsing of index, tag bits

# calculate Address parameters
address_max_size =  int(ram_size_bytes / WORD_SIZE - 1)
bits_address = int(np.log2(address_max_size+1))

# number of bits to find a word within a data block
bits_block_offset = int(np.log2(words_per_block))


# start and end of block offset bits
bit_offset_0 = 0
bit_offset_1 = bits_block_offset - 1

# number of bits used map RAM to cache
bits_index = int(np.log2(num_sets))

# start and end of index bits
bit_index_0 = bits_block_offset
bit_index_1 = bits_block_offset + bits_index - 1

# number of bits to validate data in cache = data in RAM
bits_tag = bits_address - bits_index - bits_block_offset
# start and end of tag bits
bit_tag_0 = bits_block_offset + bits_index
bit_tag_1 = bits_address - 1




# using bitmasking
address = 20

block_offset_mask = (1 << (bit_offset_1 - bit_offset_0 + 1)) - 1
index_mask = (1 << (bit_index_1 - bit_index_0 + 1)) - 1
tag_mask = (1 << (bit_tag_1 - bit_tag_0 + 1)) - 1

block_offset = address & block_offset_mask
index = (address >> bit_offset_1 + 1) & index_mask
tag = (address >> (bit_index_1 + 1)) & tag_mask

print(f"Block:{format(block_offset, 'b')}")
print(f"Index: {format(index, 'b')}")
print(f"Tag: {format(tag, 'b')}")

# using string slicing
binary = format(address, f'0{7}b')
print(binary)
rev_binary = binary[::-1]
print(f"Block: {rev_binary[bit_offset_0:bit_offset_1+1][::-1]}")
# return binary[bit_index_0:bit_index_1+1] in reversed order
print(f"Index: {rev_binary[bit_index_0:bit_index_1+1][::-1]}")
print(f"Tag: {rev_binary[bit_tag_0:bit_tag_1+1][::-1]}")
