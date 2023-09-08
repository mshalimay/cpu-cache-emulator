from data_block import DataBlock
from cpu import CPU
from cache import Cache
from ram import RAM
from address import Address
import numpy as np
import argparse
import time 
import sys

# word=double size in bytes
WORD_SIZE_BYTES = 8

def store_matrices_ram(a:list, b:list, c:list, A:np.array, B:np.array, cpu:CPU, store_sequentially:bool):
    """ Store matrices in A, B and C in RAM according to addresses in a, b and c

    Args:
        a (list): list of byte addresses for doubles of matrix A
        b (list): list of byte addresses for doubles of matrix B
        c (list): list of byte addresses for doubles of matrix C
        A (np.array): array of doubles of matrix A
        B (np.array): array of doubles of matrix A
        cpu (CPU): CPU object
        store_sequentially (bool): if True, will store first matrix A, then matrix B, then matrix C
    """
    n = A.shape[0]
    if store_sequentially:
        for i in range(n):
            for j in range(n):
                cpu.store_double(a[i][j], A[i][j])
        for i in range(n):
            for j in range(n):
                cpu.store_double(b[i][j], B[i][j])
        for i in range(n):
            for j in range(n):
                cpu.store_double(c[i][j], 0)
    else:
        for i in range(n):
            for j in range(n):
                # cache_set_A = (a[i][j] // cpu.cache.ram.data[0].size) % cpu.cache.num_sets
                # cache_set_B = (b[i][j] // cpu.cache.ram.data[0].size) % cpu.cache.num_sets
                # cache_set_C = (c[i][j] // cpu.cache.ram.data[0].size) % cpu.cache.num_sets
                cpu.store_double(a[i][j], A[i][j])
                cpu.store_double(b[i][j], B[i][j])
                cpu.store_double(c[i][j], 0)
    
        


def daxpy(n, cpu: CPU, print_results=False, alpha:float=3, print_progress=False):
    # generate range of addresses for vectors a, b, c
    a = np.array(range( 0, n*WORD_SIZE_BYTES, WORD_SIZE_BYTES))
    b = np.array(range( n*WORD_SIZE_BYTES, 2*n*WORD_SIZE_BYTES, WORD_SIZE_BYTES))
    c = np.array(range(2*n*WORD_SIZE_BYTES, 3*n*WORD_SIZE_BYTES, WORD_SIZE_BYTES))

    # Initialize some dummy values
    for i in range(n):
        cpu.store_double(byte_address=a[i], value=i)
        cpu.store_double(byte_address=b[i], value=2*i)
        cpu.store_double(byte_address=c[i], value=0)

    # put multiplication factor in register0
    register0 = alpha

    # Run the daxpy. Registers are just local variables.
    # daxpy loop: c[i] = D * a[i] + b[i]
    avg_time = 0 if print_progress else None
    for i in range(n):
        if print_progress:
            time_0 = time.time()
        # load data at address a[i] into register1
        register1 = cpu.load_double(a[i])

        # D * a[i]
        register2 = cpu.mult_double(register0, register1)

        # load data at address b[i] into register3
        register3 = cpu.load_double(b[i])
        
        # D * a[i] + b[i]
        register4 = cpu.add_double(register2, register3)

        # store result in cache/RAM at address c[i]
        cpu.store_double(c[i], register4)

        # print progress every 5% of the way
        if print_progress:
            if n//20 > 0:
                if i % (n//20) == 0:
                    print(f"{i/n*100:.0f}% of iterations done")
                    # time of iteration
                    time_diff = time.time() - time_0
                    #avg time per iteration
                    avg_time = (avg_time * (i) + time_diff)/(i+1)
                    # estimated time remaining
                    print(f"Estimated time remaining: {avg_time*(n-i)/60:.2f} minutes")
            

    if print_results:
        write_miss, read_miss, write_hit, read_hit, instructions \
            = cpu.cache.write_misses, cpu.cache.read_misses, cpu.cache.write_hits, cpu.cache.read_hits, cpu.instruction_count
        result_vector = np.array([cpu.load_double(c[i]) for i in range(n)])
        cpu.cache.write_misses, cpu.cache.read_misses, cpu.cache.write_hits, cpu.cache.read_hits, cpu.instruction_count = \
            write_miss, read_miss, write_hit, read_hit, instructions
        
        print("\n")
        print("Result vector from daxpy algorithm".center(40, "="))
        print(result_vector)
        return result_vector

    return 0

def mxm_block(A:np.array, B:np.array,cpu:CPU, block_factor:int=32, print_results = False, print_progress = False, store_sequentially=True):
    # matrix dimensions
    n = A.shape[0]

    # generate memory addresses for matrices A, B, C
    a = np.arange(0, n*n*WORD_SIZE_BYTES, WORD_SIZE_BYTES).reshape(n, n)
    b = np.arange(n*n*WORD_SIZE_BYTES, 2*n*n*WORD_SIZE_BYTES, WORD_SIZE_BYTES).reshape(n, n)
    c = np.arange(2*n*n*WORD_SIZE_BYTES, 3*n*n*WORD_SIZE_BYTES, WORD_SIZE_BYTES).reshape(n, n)  

    # store the matrices in RAM and compute the number of write misses/hits while doing so
    store_matrices_ram(a, b, c, A, B, cpu, store_sequentially)    
    write_misses_allocating_matrix  = cpu.cache.write_misses
    write_hits_allocating_matrix    = cpu.cache.write_hits

    # helper variables for printing the progress of the program
    if print_progress:
        blocks_computed = 0
        avg_time = 0
        total_num_blocks = np.ceil(n/block_factor)

    # variables for storing the number of misses/hits for each type of load/store operation
    read_misses_load_A = 0
    read_hits_load_A = 0
    
    read_misses_load_B = 0
    read_hits_load_B = 0
    
    read_misses_load_C = 0
    read_hits_load_C = 0
    
    write_misses_store_C = 0
    write_hits_store_C = 0

    # compute the block matrix multiplication
    # loop over the blocks of the matrix
    for jj in range(0, n, block_factor):
        if print_progress:
            time_0 = time.time()
        for kk in range(0, n , block_factor):

            # loop for all n rows of A blocks
            for i in range(n):
                # loops over the columns of B block
                for j in range(jj, min(jj+block_factor, n)):
                    register_0 = 0            
                    for k in range(kk, min(kk+block_factor, n)):
                        
                        miss_0, hit_0 = cpu.cache.read_misses, cpu.cache.read_hits
                        # a[i][k]
                        register_1 = cpu.load_double(a[i][k])
                        # update the number of misses/hits for loading from A
                        read_misses_load_A += cpu.cache.read_misses - miss_0
                        read_hits_load_A += cpu.cache.read_hits - hit_0

                        miss_0, hit_0 = cpu.cache.read_misses, cpu.cache.read_hits
                        # b[k][j]
                        register_2 = cpu.load_double(b[k][j])

                        # update the number of misses/hits for loading from B
                        read_misses_load_B += cpu.cache.read_misses - miss_0
                        read_hits_load_B += cpu.cache.read_hits - hit_0

                        # a[i][k] * b[k][j]    
                        register_3 = cpu.mult_double(register_1, register_2)
                        register_0 = cpu.add_double(register_0, register_3)
                        
                    miss_0, hit_0 = cpu.cache.read_misses, cpu.cache.read_hits
                    # c[i][j]
                    register_4 = cpu.load_double(c[i][j])
                    read_misses_load_C += cpu.cache.read_misses - miss_0
                    read_hits_load_C += cpu.cache.read_hits - hit_0

                    miss_0, hit_0 = cpu.cache.write_misses, cpu.cache.write_hits
                    # c[i][j] = c[i][j] + sum(a[i][k] * b[k][j])
                    cpu.store_double(c[i][j], cpu.add_double(register_0, register_4))
                    write_misses_store_C += cpu.cache.write_misses - miss_0
                    write_hits_store_C += cpu.cache.write_hits - hit_0

        if print_progress:
            # time to compute the block
            time_diff = time.time() - time_0
            blocks_computed +=1
            # average time per block
            avg_time = (avg_time * (blocks_computed-1) + time_diff)/(blocks_computed)
            
            print(f"{blocks_computed}/{total_num_blocks} of blocks done")
            print(f"Average time per block: {avg_time:.2f} seconds")
            print(f"(Estimated time remaining: {(total_num_blocks-blocks_computed+1)*avg_time/60:.0f} minutes)")

    if print_results:
        # store the state of the cache before loading the result matrix
        write_miss, read_miss, write_hit, read_hit, instructions \
            = cpu.cache.write_misses, cpu.cache.read_misses, cpu.cache.write_hits, cpu.cache.read_hits, cpu.instruction_count

        # matrix result stored in the virtual RAM
        result_matrix = np.array([[cpu.load_double(c[i][j]) for j in range(n)] for i in range(n)])

        # restore the state of the cache
        cpu.cache.write_misses, cpu.cache.read_misses, cpu.cache.write_hits, cpu.cache.read_hits, cpu.instruction_count = \
            write_miss, read_miss, write_hit, read_hit, instructions

        print("\n")
        print("Result matrix from mxm_block algorithm".center(40, "="))
        print(result_matrix)
        return result_matrix, read_misses_load_A, read_misses_load_B, read_misses_load_C, write_misses_store_C, \
            read_hits_load_A, read_hits_load_B, read_hits_load_C, write_hits_store_C, write_misses_allocating_matrix, write_hits_allocating_matrix

    return 0, read_misses_load_A, read_misses_load_B, read_misses_load_C, write_misses_store_C, \
        read_hits_load_A, read_hits_load_B, read_hits_load_C, write_hits_store_C, write_misses_allocating_matrix, write_hits_allocating_matrix

def mxm(A:np.array, B:np.array, cpu:CPU, print_results = False, print_progress = False, store_sequentially = True):
    # calculate C = A * B using pseudo-assembly operations
    n = A.shape[0]

    # generate memory addresses for matrices A, B, C
    a = np.arange(0, n*n*WORD_SIZE_BYTES, WORD_SIZE_BYTES).reshape(n, n)
    b = np.arange(n*n*WORD_SIZE_BYTES, 2*n*n*WORD_SIZE_BYTES, WORD_SIZE_BYTES).reshape(n, n)
    c = np.arange(2*n*n*WORD_SIZE_BYTES, 3*n*n*WORD_SIZE_BYTES, WORD_SIZE_BYTES).reshape(n, n)  

    # store matrices in RAM memory and compute the number of misses/hits while doing so
    store_matrices_ram(a, b, c, A, B, cpu, store_sequentially)
    write_misses_allocating_matrix  = cpu.cache.write_misses
    write_hits_allocating_matrix = cpu.cache.write_hits
    

    # variables to store the number of misses/hits for each operation
    read_misses_load_A = 0
    read_hits_load_A = 0
    read_misses_load_B = 0
    read_hits_load_B = 0
    write_misses_store_C = 0
    write_hits_store_C = 0

    avg_time = 0 if print_progress else None
    # matrix multiplication loop
    # loop over rows
    for i in range(n):
        if print_progress:
            time_0 = time.time()
        # loop over columns
        for j in range(n):
            register_0 = 0
            for k in range(n):
                miss_0, hit_0 = cpu.cache.read_misses, cpu.cache.read_hits
                # a[i][k]
                register_1 = cpu.load_double(a[i][k])
                # compute the number of misses/hits for loading from A
                read_misses_load_A += cpu.cache.read_misses - miss_0
                read_hits_load_A += cpu.cache.read_hits - hit_0

                miss_0, hit_0 = cpu.cache.read_misses, cpu.cache.read_hits
                # b[k][j]
                register_2 = cpu.load_double(b[k][j])
                read_misses_load_B += cpu.cache.read_misses - miss_0
                read_hits_load_B += cpu.cache.read_hits - hit_0
               
                # a[i][k] * b[k][j]
                register_3 = cpu.mult_double(register_1, register_2)

                # c = c + a[i][k] * b[k][j]
                register_0 = cpu.add_double(register_0, register_3)

            miss_0, hit_0 = cpu.cache.write_misses, cpu.cache.write_hits
            # c[i][j] = sum(a[i][k] * b[k][j])
            cpu.store_double(c[i][j], register_0)
            # compute the number of misses/hits for storing to C
            write_misses_store_C += cpu.cache.write_misses - miss_0
            write_hits_store_C += cpu.cache.write_hits - hit_0

        if print_progress:
            if n//20 > 0:
                if i % (n//20) == 0:
                    time_diff = time.time() - time_0
                    # average time per iteration
                    avg_time = (avg_time * (i) + time_diff)/(i+1)
                    # print iterations completed, average time per iteration, and estimated time remaining
                    print(f"{i+1}/{n} of iterations done")
                    print(f"Average time per iteration: {avg_time:.2f} seconds")
                    print(f"(Estimated time remaining: {(n-i-1)*avg_time/60:.0f} minutes)")        
        
    if print_results:
        # store the state of the cache before load result matrix from virtual RAM
        write_miss, read_miss, write_hit, read_hit, instructions \
            = cpu.cache.write_misses, cpu.cache.read_misses, cpu.cache.write_hits, cpu.cache.read_hits, cpu.instruction_count

        # load result matrix from virtual RAM
        result_matrix = np.array([[cpu.load_double(c[i][j]) for j in range(n)] for i in range(n)])

        # restore the state of the cache
        cpu.cache.write_misses, cpu.cache.read_misses, cpu.cache.write_hits, cpu.cache.read_hits, cpu.instruction_count = \
            write_miss, read_miss, write_hit, read_hit, instructions

        print("\n")
        print("Result matrix from mxm algorithm".center(40, "="))
        print(result_matrix)
        return result_matrix, read_misses_load_A, read_misses_load_B, write_misses_store_C, write_misses_allocating_matrix, \
            read_hits_load_A, read_hits_load_B, write_hits_store_C, write_hits_allocating_matrix

    return 0, read_misses_load_A, read_misses_load_B, write_misses_store_C, write_misses_allocating_matrix, \
        read_hits_load_A, read_hits_load_B, write_hits_store_C, write_hits_allocating_matrix

def print_config(pseudo_ram_size:int, ram_size_bytes:int, cache_size_bytes:int, data_block_size_bytes:int, 
                 num_blocks:int, n_way_associativity:int, cpu:CPU, repl_policy:str, algorithm:str, block_factor:int, n:int):
    """ Print configurations for the current run of the program"""
    
    if algorithm == "mxm_block":
        algorithm = "mxm blocked"
    else:
        block_factor = "Not applicable"
        
    print("INPUTS".center(40, "=")
    + f"\nRequired RAM Size = {pseudo_ram_size} bytes"
    + f"\nEffective Ram Size = {ram_size_bytes} bytes"
    + f"\nCache Size = {cache_size_bytes} bytes"
    + f"\nBlock Size = {data_block_size_bytes} bytes"
    + f"\nTotal Blocks in Cache = {num_blocks}"
    + f"\nAssociativity = {n_way_associativity}"
    + f"\nNumber of Sets = {cpu.cache.num_sets}"
    + f"\nReplacement Policy = {repl_policy}"
    + f"\nAlgorithm = {algorithm}"
    + f"\nMXM Blocking Factor = {block_factor}"
    + f"\nMatrix or Vector dimension = {n}")

def print_stats(cpu:CPU):
    """ Print statistics for the current run of the program"""
    read_miss_rate = cpu.cache.read_misses / (cpu.cache.read_hits + cpu.cache.read_misses) * 100
    write_miss_rate = cpu.cache.write_misses / (cpu.cache.write_hits + cpu.cache.write_misses) * 100

    print(f"RESULTS".center(40, "="))
    print(f"Instruction count: {cpu.instruction_count}"
    + f"\nRead hits: {cpu.cache.read_hits}"
    + f"\nRead misses: {cpu.cache.read_misses}"
    + f"\nRead miss rate: {read_miss_rate:.2f}%"
    + f"\nWrite hits: {cpu.cache.write_hits}"
    + f"\nWrite misses: {cpu.cache.write_misses}"
    + f"\nWrite miss rate: {write_miss_rate:.2f}%")


#=================================================================================================
# Main
#=================================================================================================
if __name__ == "__main__":

    #=================================================================================================
    # Parsing and processing command line arguments
    #s=================================================================================================

    # instantiate parser and define command line arguments, help messages, default values
    parser = argparse.ArgumentParser(description='Cache emulator')
    parser.add_argument('-c', '--cache_size', dest='cache_size_bytes', type=int, default=65536, help='The size of the cache in bytes (default: 65,536)', metavar='')
    parser.add_argument('-b', '--block_size', dest='data_block_size_bytes', type=int, default=64, help='The size of a data block in bytes (default: 64)', metavar='')
    parser.add_argument('-n', '--associativity', dest='n_way_associativity', type=int, default=2,help='The n-way associativity of the cache. -n 1 is a direct-mapped cache. (default: 2)', metavar='')
    parser.add_argument('-r', '--replacement_policy', dest='replacement_policy', type=str, default='LRU',help='The replacement policy. Can be random, FIFO, or LRU. (default: LRU)', metavar='')
    parser.add_argument('-a', '--algorithm', dest='algorithm', type=str, default='mxm_block', help='The algorithm to simulate. Can be daxpy (daxpy product), mxm (matrix-matrix multiplication), mxm block (mxm with blocking). (default: mxm block).', metavar='')
    parser.add_argument('-d', '--dimension', dest='n', type=int, default=480, help='The dimension of the algorithmic matrix (or vector) operation. -d 100 would result in a 100 × 100 matrix-matrix multiplication. (default: 480)', metavar='')
    parser.add_argument('-p', '--print_solution', dest='print_results', action='store_true', help='Enables printing of the resulting “solution” matrix product or daxpy vector after the emulation is complete.')
    parser.add_argument('-f', '--blocking_factor', dest='block_factor', type=int, default=32, help='The blocking factor for use when using the blocked matrix multiplication algorithm. (default: 32)', metavar='')
    parser.add_argument('-pp', '--print_progress', dest='print_progress', action='store_true', help='Enables printing of progress during emulation.')
    parser.add_argument('-breakdown', dest='breakdown_hits_misses', action='store_true', help='Enables printing of misses and hits breakdown.')
    parser.add_argument('-store_seq', dest='store_seq', action='store_true', help='Store matrices one after the other (such as A,B,C order.')
    

    # if invalid arguments, print help message and exit
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as err:
        parser.print_help()
        sys.exit(2)

    if args.replacement_policy not in ['random', 'FIFO', 'LRU']:
        print("Invalid replacement policy. Must be random, FIFO, or LRU.")
        sys.exit(2)

    if args.algorithm not in ['daxpy', 'mxm', 'mxm_block']:
        print("Invalid algorithm. Must be daxpy, mxm, or mxm_block.")
        sys.exit(2)

    if args.n_way_associativity < 1:
        print("Invalid associativity. Please choose a power of 2 >= 1.")
        sys.exit(2)

    if args.data_block_size_bytes < WORD_SIZE_BYTES:
        print(f"Invalid data block size {args.data_block_size_bytes} bytes. Must be at least {WORD_SIZE_BYTES} bytes.")
        sys.exit(2)

    if np.log2(args.n_way_associativity) % 1 != 0:
        print(f"WARNING: Invalid associativity {args.n_way_associativity}. Must be a power of 2.")
        n_way_associativity = int(2**np.ceil(np.log2(args.n_way_associativity)))
        print(f"Associativity set to the next power of 2: {n_way_associativity}\n")
    
    # put command line arguments into local variables for ease of use
    locals().update(vars(args))
 
    #=================================================================================================
    # calculate RAM size and preprocess cache size
    #=================================================================================================
    
    # calculate RAM size:
    #  pseudo-ram size is the amount of RAM that would be required to store the data
    #  effective RAM size is the first power of 2 that is larger than the pseudo-ram size
    if algorithm == "daxpy":
        pseudo_ram_size = 3*n*WORD_SIZE_BYTES
    elif algorithm == "mxm" or algorithm == "mxm_block":
        pseudo_ram_size = 3*n**2*WORD_SIZE_BYTES

    ram_size_bytes = int(2**np.ceil(np.log2(pseudo_ram_size)))

    if ram_size_bytes <= data_block_size_bytes:
        print(f"ERROR: RAM size {ram_size_bytes} is too small. Must be larger than data block size {data_block_size_bytes}")
        sys.exit(2)
    
    # if cache size is larger than effective RAM size, set cache size to half of effective RAM size
    if ram_size_bytes < cache_size_bytes:
        print(f"WARNING: Cache size {cache_size_bytes} is larger than effective RAM size {ram_size_bytes}.")
        cache_size_bytes = ram_size_bytes // 2
        print(f"Cache size will be set to effective_ram_size / 2: {cache_size_bytes} bytes\n")
        
    #=================================================================================================
    # Instantiate CPU, RAM, Cache, and Address
    #=================================================================================================

    # instantiate RAM
    words_per_block = data_block_size_bytes // WORD_SIZE_BYTES
    num_ram_blocks = ram_size_bytes // data_block_size_bytes

    data_blocks = [DataBlock(size=data_block_size_bytes, data=[0.0] * words_per_block) for _ in range(num_ram_blocks)]
    ram = RAM(num_blocks=num_ram_blocks, data = data_blocks)

    # instantiate Cache
    num_cache_blocks = cache_size_bytes // data_block_size_bytes
    cache = Cache(n_way_associativity = n_way_associativity, num_blocks = num_cache_blocks, ram = ram, repl_policy = replacement_policy)

    # calculate Address parameters and instantiate it
    max_word_address = ram_size_bytes // WORD_SIZE_BYTES - 1
    bits_for_word_address = int(np.log2(ram_size_bytes / WORD_SIZE_BYTES))
    bits_for_block_offset = int(np.log2(words_per_block))
    bits_for_index = int(np.log2(cache.num_sets))

    address =  Address(byte_address=0, bits_for_word_address = bits_for_word_address, 
                    bits_for_block_offset=bits_for_block_offset, bits_for_index=bits_for_index, 
                    word_size_bytes=WORD_SIZE_BYTES)


    # instantiate the CPU
    cpu = CPU(cache = cache, address = address)

    #=================================================================================================
    # Run requested algorithm and print solution
    #=================================================================================================
    if algorithm == "daxpy":
        result_vector = daxpy(n, cpu, print_results=print_results, alpha=3, print_progress=print_progress)
        if print_results:
            print("\n")
            print("Result vector from daxpy using numpy".center(35, "="))
            a = np.array(range(n))
            b = 2*a
            numpy_result = 3*a + b
            print(numpy_result)

            print("\n Solution - Numpy daxpy (expected: zeros)".center(20, "="))
            print(result_vector - numpy_result)

    elif algorithm == "mxm" or algorithm == "mxm_block":
        a = np.array(range(n**2)).reshape(n, n)
        b = 2*a
        if algorithm == "mxm":
            result_matrix, read_misses_load_A, read_misses_load_B, write_misses_store_C, write_misses_allocating_matrix, \
            read_hits_load_A, read_hits_load_B, write_hits_store_C, write_hits_allocating_matrix \
                = mxm(A=a, B=b, cpu=cpu, print_results=print_results, print_progress=print_progress, store_sequentially=store_seq)
        else:
            result_matrix, read_misses_load_A, read_misses_load_B, read_misses_load_C, write_misses_store_C, \
            read_hits_load_A, read_hits_load_B, read_hits_load_C, write_hits_store_C, write_misses_allocating_matrix, \
                write_hits_allocating_matrix \
                    = mxm_block(A=a, B=b, cpu=cpu, print_results=print_results, block_factor=block_factor, print_progress=print_progress, store_sequentially=store_seq)    

        if print_results:
            print("\n")
            print("Result matrix using numpy multiplication".center(30, "="))
            numpy_result = a@b
            print(numpy_result)

            print("\n Solution - Numpy multiplication (expected: zeros)".center(10, "="))
            print(result_matrix - a@b)

    #=================================================================================================
    # Print configuration and statistics 
    #=================================================================================================
    print("\nConfiguration and Statistics")
    print_config(pseudo_ram_size = pseudo_ram_size, ram_size_bytes=ram_size_bytes, cache_size_bytes=cache_size_bytes, data_block_size_bytes=data_block_size_bytes, 
                 num_blocks=num_cache_blocks, n_way_associativity=n_way_associativity, cpu=cpu, repl_policy=replacement_policy,
                 algorithm=algorithm, block_factor=block_factor, n=n)
    print_stats(cpu)

    # print finer breakdown of hits and missesfor matrix mult algorithms
    if breakdown_hits_misses:
        print("\nBreakdown of misses")
        print("Write misses: ", cpu.cache.write_misses)
        print("Write misses allocating matrices: ", write_misses_allocating_matrix)
        print("Write misses store C: ", write_misses_store_C)

        print("\n")
        print("Read misses: ", cpu.cache.read_misses)
        print("Read misses load A: ", read_misses_load_A)
        print("Read misses load B: ", read_misses_load_B)
        if algorithm=="mxm_block":
            print("Read misses load C: ", read_misses_load_C)

        print("\nBreakdown of hits")
        print("Write hits: ", cpu.cache.write_hits)
        print("Write hits allocating matrices: ", write_hits_allocating_matrix)
        print("Write hits store C: ", write_hits_store_C)

        print("\n")
        print("Read hits: ", cpu.cache.read_hits)
        print("Read hits load A: ", read_hits_load_A)
        print("Read hits load B: ", read_hits_load_B)
        if algorithm=="mxm_block":
            print("Read hits load C: ", read_hits_load_C)
