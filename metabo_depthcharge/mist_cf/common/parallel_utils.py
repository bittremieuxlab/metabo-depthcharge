# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""parallel_utils.py"""


def simple_parallel(input_list, function, max_cpu=16, timeout=4000, max_retries=3):
    from pathos import multiprocessing as mp

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)
    results = pool.map(function, input_list)
    pool.close()
    pool.join()
    return results


def chunked_parallel(
    input_list, function, chunks=100, max_cpu=16, timeout=4000, max_retries=3
):
    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    list_outputs = simple_parallel(
        chunked_list,
        batch_func,
        max_cpu=max_cpu,
        timeout=timeout,
        max_retries=max_retries,
    )
    full_output = [j for i in list_outputs for j in i]
    return full_output
