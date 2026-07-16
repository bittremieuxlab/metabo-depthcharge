# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""parallel_utils.py"""


def simple_parallel(input_list, function, max_cpu=16, timeout=4000, max_retries=3, pbar=False, desc=None):
    from pathos import multiprocessing as mp

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)
    if pbar:
        from tqdm import tqdm

        tagged = list(enumerate(input_list))

        def _run_tagged(item, _fn=function):
            i, x = item
            return i, _fn(x)

        pairs = list(tqdm(pool.imap_unordered(_run_tagged, tagged), total=len(input_list), desc=desc))
        pairs.sort(key=lambda p: p[0])
        results = [r for _, r in pairs]
    else:
        results = pool.map(function, input_list)
    pool.close()
    pool.join()
    return results


def chunked_parallel(
    input_list, function, chunks=100, max_cpu=16, timeout=4000, max_retries=3, pbar=False
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
        pbar=pbar,
    )
    full_output = [j for i in list_outputs for j in i]
    return full_output
