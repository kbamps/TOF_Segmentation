import concurrent.futures
import sys
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def parallel_tasks(function, arguments_list, num_workers=1, description="processing", show_bar=True, force_single_thread=False):
    """
    Executes a function in parallel using multiple worker processes.
    
    Args:
        function (callable): The function to execute in parallel.
        arguments_list (list): A list of argument tuples to pass to the function.
        num_workers (int, optional): Number of workers to use for parallel execution. Defaults to 1.
        show_bar (bool, optional): Whether to display a progress bar using tqdm. Defaults to True.
        force_single_thread (bool, optional): Forces single-threaded execution, useful for debugging.
    
    Returns:
        list: A list of results from the parallel execution, in the same order as the input argument list.
    """
    force_single_thread = force_single_thread
    disabled = not show_bar
    results_list = [None] * len(arguments_list)  # Preallocate the result list
    total_tasks = len(arguments_list)
    num_workers = min(len(arguments_list), num_workers or 1)
    
    with tqdm(total=total_tasks, desc=description, unit="item", disable=disabled) as pbar:
        if not force_single_thread:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(function, *args): idx for idx, args in enumerate(arguments_list)}
                
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]  # Get the index for the completed future
                    try:
                        results_list[idx] = future.result()  # Store the result at the correct index
                    except Exception as e:
                        logger.info(f"Unable to get result for task {idx}-{arguments_list[idx]}: {e}")
                    
                    pbar.update(1)
        else:
            for idx, args in enumerate(arguments_list):
                try:
                    results_list[idx] = function(*args)
                except Exception as e:
                    logger.info(f"Unable to get result for task {idx}-{arguments_list[idx]}: {e}")

                pbar.update(1)

    return results_list