from torch import Tensor

from planet.data import PlaNetDataset


def test_planet_dataset():
    ds = PlaNetDataset(path="planet/tests/data/iter_like_data_sample.h5")

    assert hasattr(ds, "RR")
    assert hasattr(ds, "ZZ")
    assert hasattr(ds, "flux")
    assert hasattr(ds, "rhs")
    assert hasattr(ds, "inputs")

    data = ds[0]
    assert isinstance(data, list)

    for x in data:
        assert isinstance(x, Tensor)
