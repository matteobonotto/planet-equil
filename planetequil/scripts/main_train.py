from planetequil.utils import parse_arguments, load_config
from planetequil.train import main_train


if __name__ == "__main__":

    # load config
    args = parse_arguments()
    config = load_config(args.config)

    # train and save the model
    main_train(config=config)
