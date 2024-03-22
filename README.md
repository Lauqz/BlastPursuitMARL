# BlastPursuit

## Overview
BlastPursuit is a Python-based project designed for training and evaluating reinforcement learning agents in a gaming scenario. It provides tools for training agents and evaluating their performance using a predefined environment. The project includes a single agent implementation for users to experiment with.

## Installation
To use BlastPursuit, follow these steps:

- Clone the repository:
```
cd repo_folder
git clone https://github.com/Lauqz/BlastPursuitMARL.git
```

- Install the required packages using pip:
```
pip install -r requirements.txt
```

## Usage
- Training: Launch the training process using the following command:

```
python training.py
```

- Evaluation: Evaluate the trained agent by running the evaluation script:

```
python evaluate.py
```

## Additional Notes
- Convergence Monitoring: Monitor the training convergence using TensorBoard. Run the following command in the project directory:

```
tensorboard --logdir tensorboard
```

- GUI Modification: To view the GUI during training, modify the utils.py file. Change the value of TRAINING from True to False.

## Authors and acknowledgment
Authors: Guido Laudenzi and Sami Osman.
Thanks to the contributors of the open-source libraries used in this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

