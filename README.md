# Autopilot-TensorFlow-BNN
A TensorFlow implementation of this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) with some changes. This repo tries to test whether it will still work after binarizing all the weights and input features.

This implentation is based on [SullyChen's repo](https://github.com/SullyChen/Autopilot-TensorFlow)

# How to Use
Download the [dataset](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing) and extract into the repository folder

Use `python train.py` to train the model

Use `python run.py` to run the model on a live webcam feed

Use `python run_dataset_autonomy.py` to run the model on the dataset and check the autonomy(mentioned in the Nvidia paper)

To visualize training using Tensorboard use `tensorboard --logdir=./logs`, then open http://0.0.0.0:6006/ into your web browser.
