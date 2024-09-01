# Super Mario Bros. PPO Agent

This project trains a PPO-based agent that can speedrun the Super Mario Bros. (NES). After training with `train.py`, `render.py` can use the trained model to play level 4-1 and render the gameplay process as an MP4 file. The project also provides behavior cloning training, which can be started by `imitation.py`.

## Requirements

- Python 3.10 or higher
- A C++ compiler that supports C++17

The project has been tested on the following environments:

- Python 3.11 on Apple M1 chip
- Python 3.10 on AMD Ryzen + RTX 4070 (Linux)

## Installation

Before running the programes, please install the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Train a PPO agent

Before training the agent, please use the following code to create a working directory first.

```bash
python3 train.py -c <working_directory>
```

The code will create a directory with a configuration file `config.json` contains default settings for the training. Please modify the configuration file to fit your needs before the training.

The next step is to start the training:

```bash
python3 train.py <working_directory>
```

The code can automatically select the device to train the agent. It will also start a tensorboard server listening on `localhost:7007`. 

The following is the essential objects in working directory:

- `config.json`: The configuration file for the training.
- `logs`: The directory contains the tensorboard logs.
- `checkpoints`: The directory contains the saved models during the training.
- `model.zip`: The final version of the model produced by a successful training.
- `base_model.zip`: If this saved model file exists, the training programme will load the model at the begining, instead of creating new model.

## Evaluate the trained agent

To evaluate the trained model, please use the following code to run the agent within the environment:

```bash
python3 render.py -s <frame_skipping_number> <model_file>
```

For example, to evaluate the model in the working directory `work/default_cnn` with frame skipping set to `3`, you can use the following code:

```bash
python3 render.py -s 3 work/default_cnn/model.zip
```

The `render.py` produces an MP4 file along with the specified saved model. 

## Imitation learning

The project also provides a Behaviour Cloning training programme. To start the training, please use the following code:

```bash
python3 imitate.py --epoches <epoches> --batch_size <batch_size> <expert_data_dir> <working_directory>
```

The programme trains a new model with the expert gameplay data from the `expert_data_dir` and outputs the trained model as `<working_directory>/base_model.zip`, which can be used by the `train.py`.

