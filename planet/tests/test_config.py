import yaml

from planet.config import Config, PlaNetConfig


def test_config():

    #
    cfg = yaml.safe_load(open("planet/tests/data/config.yml"))
    config = Config.from_dict(cfg)

    #
    config = PlaNetConfig(
        batch_size=128,
        branch_in_dim=10000,
    )

test_config()