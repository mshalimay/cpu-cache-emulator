import numpy as np
from data_block import DataBlock
from address import Address
from ram import RAM
from collections import OrderedDict
import random

class Cache:
    # constructor
    def __init__(self, n_way_associativity:int , num_blocks:int, ram:RAM, repl_policy:str="LRU"):

        self.num_blocks = num_blocks
        self.num_sets = self.num_blocks // n_way_associativity
        self.ram = ram
        self.repl_policy = repl_policy

        # initialize the cache blocks
        # Obs.: blocks is a List(OrderedDict[int, DataBlock])
        self.sets = [OrderedDict((-i, None) for i in range(1, n_way_associativity+1)) for _ in range(self.num_sets)]
        
        # counters for cache hits and misses
        self.read_hits = 0
        self.read_misses = 0
        self.write_hits = 0
        self.write_misses = 0
        self.read_compulsory_misses = 0
        self.write_compulsory_misses = 0

    def get_double(self, address:Address) -> float:
        """ Gets the double at the given address from the cache. 
        If the double is not in the cache, it is fetched from RAM and inserted into the cache
        Counters for cache hit/miss are updated accordingly

        Args:
            address (Address): an address object representing the word address where the double is located in RAM
            notice: not the byte address; word address = byte address / size of double in bytes

        Returns:
            float: the double at the given address
        """

        # compute the index and tag from the address
        index = address.get_index()
        tag = address.get_tag()

        # get the set from the cache 
        cache_set = self.sets[index]

        # look for a datablock in the set with the given tag
            # Note: no need for the valid bit; no tag will be a valid tag (i.e., an integer >= 0)
            # in the initial state of the cache
        if tag in cache_set:
            # if found the data block => cache hit, update the set order according to the replacement policy
            data_block = self.update_set_order(cache_set, tag)
            self.read_hits += 1
        else:
            # if tag is not found in the set => cache miss, get the data block from RAM and insert into the cache
            data_block = self.get_block(address.address)
            self.read_compulsory_misses += self.insert_block_into_cache(data_block, cache_set, tag)
            self.read_misses += 1

        # return the double from the data block at the offset given by the address
        return data_block.get_double(address.get_block_offset())
 
    def get_block(self, word_address:int) -> DataBlock:
        """ Gets the data block at the given word address from RAM"""
        ram_data_block = self.ram.get_block(word_address)
        return ram_data_block

    def set_double(self, address:Address, value:float) -> None:
        """ Sets a value to the double at the given address in cache. 
        If the double is not in the cache, it is fetched from RAM, 
        inserted into the cache, updated then written back to RAM
        Counters for cache hit/miss are updated accordingly

        Args:
            address (Address): an address object representing the word address where the double is located in RAM
            notice: not the byte address; word address = byte address / size of double in bytes
        """

        # compute the index and tag from the address
        index = address.get_index()
        tag = address.get_tag()

        # get the set from the cache
        cache_set = self.sets[index]


        # look for a datablock in the set with the given tag
            # Note: no need for the valid bit; no tag will be a valid tag (i.e., an integer >= 0)
            # in the initial state of the cache
        if tag in cache_set:
            #  if found the data block => cache hit, update the set order according to the replacement policy
            data_block = self.update_set_order(cache_set, tag)
            self.write_hits += 1
        else:
            # if cache miss, get the data block from RAM and insert into the cache
            data_block = self.get_block(address.address)
            self.write_compulsory_misses += self.insert_block_into_cache(data_block, cache_set, tag)
            self.write_misses += 1

        # write the double value into the cache data block
        data_block.set_double(address.get_block_offset(), value)

        # write the data block to RAM
        self.set_block(address.address, data_block)

    
    def set_block(self, word_address:int, data_block:DataBlock) -> None:
        """ Writes the data block to RAM at the given word address"""
        self.ram.set_block(word_address, data_block)
               
    def insert_block_into_cache(self, ram_data_block:DataBlock, cache_set:OrderedDict, tag:int) -> int:
        """ Inserts a data block into the cache  according to the replacement policy"""

         # note: using OrderedDict, it does not matter if the cache is direct mapped;
         # replacement pattern is the same for an associative cache as long as the 
         # order of items in the set is correct (i.e., from LRU --> MRU)
         
        if self.repl_policy == "LRU" or self.repl_policy =="FIFO":
            # pop the first item in the set = LRU
            # add the new item to the end of the set = MRU
            #   eg: insert DataBlock17
            #   Initial cache set: LRU ---> DataBlock1 -> DataBlock3 -> DataBlock2 ---> MRU
            #   After 1st pop:     LRU ---> DataBlock3 -> DataBlock2 ---> MRU
            #   After insert:      LRU ---> DataBlock3 -> DataBlock2 -> DataBlock17 ---> MRU
            key, value = cache_set.popitem(last=False)
            cache_set[tag] = ram_data_block
            if key < 0:
                return 1
            return 0

        elif self.repl_policy == "random":
            # select random key in cache_set and pop from it
            random_tag = random.choice(list(cache_set.keys()))
            cache_set.pop(random_tag)
            cache_set[tag] = ram_data_block
            if random_tag < 0:
                return 1
            return 0
        
        
    def update_set_order(self, set, tag):
        if self.repl_policy == "LRU":
            # order the set from LRU --> MRU every time a cache hit occurs
            # this way, the set is always in FIFO order
            data_block = set.pop(tag)
            set[tag] = data_block
            return data_block

        elif self.repl_policy == "FIFO":
            return set[tag]

        elif self.repl_policy == "random":
            return set[tag]

        else:
            # direct mapped case
            return set[tag]