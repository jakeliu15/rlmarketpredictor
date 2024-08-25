# rltradebot

# FinRL Stock Trading Project

This project implements a reinforcement learning-based stock trading system using the FinRL library. It trains and evaluates various RL algorithms for automated stock trading.

## Features

- Utilizes the FinRL library for stock trading environment setup
- Implements multiple RL algorithms including A2C and PPO
- Trains models on historical stock data
- Saves trained models for future use

## Prerequisites

- Python 3.7+
- pandas
- numpy
- matplotlib
- FinRL
- Stable-Baselines3

## Installation

1. Clone this repository
2. Install required packages:
   pip install pandas numpy matplotlib finrl stable-baselines3

## Usage

1. Prepare your training data in a CSV file named 'train_data.csv'
2. Run the main script:
   python main.py
3. The script will train the selected models (A2C and PPO by default) and save them in the 'trained_models' directory

## Configuration

- Modify the `if_using_*` variables to select which algorithms to train
- Adjust `total_timesteps` in the `agent.train_model()` calls to change training duration
- Modify `env_kwargs` to adjust trading environment parameters

## Results

- Training logs and TensorBoard files are saved in the 'results' directory
- Trained models are saved in the 'trained_models' directory

## Note

This project is for educational purposes only. Always consult with a financial advisor before making investment decisions.
