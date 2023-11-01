# LSTM-based Sales Forecasting

## Table of Contents

1. [Introduction](#Introduction)
2. [Key Components](#Key-Components)
3. [Code Details](#Code-Details)
4. [Installation and Setup](#Installation-and-Setup)
5. [Usage](#Usage)
6. [Evaluation Metrics](#Evaluation-Metrics)

## Introduction

This project aims to predict future sales using a recurrent neural network (RNN) architecture called Long Short-Term Memory (LSTM). It is a complete, self-contained example covering various steps, from data creation and pre-processing to model building, training, and evaluation.

## Key Components

### Data Generation

Synthetic sales data for 1000 days is generated using a sine function for seasonality, a trend component, and random noise.

### Data Preprocessing

1. **Loading Data**: The generated data is loaded into a Pandas DataFrame.
2. **Scaling**: The `Sales` column is scaled between 0 and 1 using MinMaxScaler.
3. **Sequence Preparation**: Sequences of length `seq_len=10` are created to train the LSTM model. Each sequence is used to predict the next sales value.

### Train/Test Split

80% of the data is used for training, and the rest is used for testing.

### Custom Callback

This is used for printing batch end logs.

### TensorBoard

Used for monitoring training in real-time.

### LSTM Model

1. The first LSTM layer with 50 units returns sequences to be used by the next LSTM layer.
2. The second LSTM layer also has 50 units but doesn't return sequences.
3. A Dense layer with a single unit is used for the output.

## Code Details

- **CustomCallback Class**: Inherits from `Callback` to customize behavior during batch training. It prints the loss after each batch ends.
- **TensorBoard**: Logs are stored in the `./logs` directory and can be visualized using TensorBoard.
- **LSTM Layers**: Two LSTM layers are stacked to capture more complex patterns in the data.
- **Model Compilation and Training**: The `adam` optimizer and `mse` loss function are used.
- **Evaluation**: The model's predictions are inverse transformed and compared against the real values to calculate the MAE.

## Installation and Setup

To run this project, clone the repository and install the necessary packages.

```bash
git clone https://github.com/your-username/LSTM-Sales-Forecasting.git
cd LSTM-Sales-Forecasting
pip install -r requirements.txt
