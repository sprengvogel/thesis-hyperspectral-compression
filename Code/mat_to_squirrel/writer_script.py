from mat_to_squirrel.dataset_helper import *
from mat_to_squirrel.squirrel_ext import ConfigurableMessagePackDriver
from squirrel.iterstream import IterableSource
from squirrel.store.filesystem import get_random_key
from functools import partial
from fnc import compose
from rich.progress import track
import rich_click.typer as typer

def main(
    output_path: Path = "/data/datasets/fatih/squirrel/full/",
    mat_path: Path = "/data/datasets/fatih/crop/data/",
    shard_size: int = 64,
    reorder_np_axis_to_torch_style: bool = True,
):
    """
    Currently using two async_maps to further reduce the compute time from 8min to 3min30sek for the dummy dataset.
    BUT I don't know if they items within the batch are in-order

    I got very weird behaviour when executing the IterableStream async_map function from within a Jupyter Notebook.
    The issues was that the data was written with the 'correct' speed, but the loop didn't continue (all data was written
    and never re-written) and I had to wait another 25min until the loop progressed and then the loop progressed very fast.
    I don't know what was causing the synchronisation issue.
    """

    paths = get_mat_paths(mat_path)
    d = random_data_split(paths, train_split_perc=0.70, validation_split_perc=0.20, test_split_perc=0.10)
    assert len(d.train) + len(d.test) + len(d.validation) == len(paths)

    msgpack_driver = ConfigurableMessagePackDriver(str(output_path))

    # always ensure that the directory is empty!
    assert len(list(msgpack_driver.keys())) == 0

    def msgpack_prefix_writer(batch, prefix: str):
        random_suffix = get_random_key()
        key = f"{prefix}_{random_suffix}"
        msgpack_driver.store.set(batch, key=key)

    splits = list(Split)
    splits = splits[::-1]
    for split in track(splits):
        print("Start of:")
        print(split)
        dataset_split_paths = getattr(d, split)
        print("Len of dataset_split_paths: ", len(dataset_split_paths))
        our_mat_path_to_np = partial(mat_path_to_np, array_key="img")
        msgpack_split_writer = partial(msgpack_prefix_writer, prefix=split)
        path_to_np = compose(our_mat_path_to_np, np_to_torch_tns_layout) if reorder_np_axis_to_torch_style else our_mat_path_to_np
        (
            IterableSource(dataset_split_paths)
            .async_map(path_to_np)
            .batched(shard_size, drop_last_if_not_full=False)
            .tqdm()
            .async_map(msgpack_split_writer, buffer=10)
            .join()
        )
        print("End of:")
        print(split)

if __name__ == "__main__":
    typer.run(main)
