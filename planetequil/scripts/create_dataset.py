from argparse import Namespace, ArgumentParser
from iterlike_equil_dataset import IterlikeDataset
from tqdm import tqdm
from pathlib import Path

from planetequil.utils import write_h5, read_h5_numpy


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--sample", action="store_true", default=True, help="Use the full dataset"
    )
    parser.add_argument("--path", default=".", help="Use the full dataset")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    dataset_id = (
        "matteobonotto/iterlike-equil-sample"
        if args.sample
        else "matteobonotto/iterlike-equil"
    )
    print(f"Loading huggingface dataset {dataset_id}")

    filename = "iter_like_data"
    if args.sample:
        filename += "_sample"
    full_path = Path(args.path) / Path(filename)

    # load huggingface dataset
    dataset = IterlikeDataset(dataset_id=dataset_id)

    # convert dataset to dictionary
    print("assemgbing dataset ...")
    data = {}
    pbar = tqdm(dataset.equil_data.column_names, total=dataset.equil_data.num_columns)
    for key in pbar:
        pbar.set_description(f"Processing {key}")
        data[key] = dataset.equil_data[key]

    # store the data in h5 dataset
    print(f"Saving data to {filename}.h5")
    write_h5(
        data=data,
        filename=str(full_path),
    )
