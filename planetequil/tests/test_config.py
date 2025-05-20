import yaml

from planetequil.config import Config, PlaNetConfig


def test_config():

    #
    cfg = yaml.safe_load(open("planetequil/tests/data/config.yml"))
    config = Config.from_dict(cfg)

    #
    config = PlaNetConfig(**config.planet.to_dict())
