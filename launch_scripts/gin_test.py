import fire
import gin


@gin.configurable
def other_function(test="jc", layers=(1, 2, 3)):
    print("Other function", test)
    print("Layers", layers)


@gin.configurable
def train(batch_size=32, learning_rate=0.001, epochs=10):
    print(f"Training with batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")


def main(config_file):
    gin.parse_config_file(config_file)
    print(gin.config_str())

    train()
    other_function()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config_file", type=str, default="config/config.gin")
    args = parser.parse_args()
    main(args.config_file)
