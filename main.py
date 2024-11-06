import argparse
import os
import math, time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import plotly.graph_objects as go

from models.Nets import LSTM, GRU
from utils.parameters import args_parse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data = []
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]
    return [x_train, y_train, x_test, y_test]

def main():
    # Parse arguments
    args = args_parse()

    # Ensure checkpoints directory exists
    if not os.path.exists('/root/autodl-tmp/stock_prediction/checkpoints'):
        os.makedirs('/root/autodl-tmp/stock_prediction/checkpoints')

    # Load and process data
    filepath = '/root/autodl-tmp/stock_prediction/data/amazon-stock-price/AMZN_data_1999_2022.csv'

    data = pd.read_csv(filepath)
    data = data.sort_values('Date')
    
    # Plot stock price
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 9))
    plt.plot(data[['Close']])
    plt.xticks(range(0, data.shape[0], 500), data['Date'].loc[::500], rotation=45)
    plt.title("Amazon Stock Price", fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price (USD)', fontsize=18)
    plt.show()

    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price = data[['Close']]
    price.loc[:, 'Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))

    
    # Split data
    lookback = args.lookback
    x_train, y_train, x_test, y_test = split_data(price, lookback)
    
    # Convert to PyTorch tensors
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # Get model
    if args.model == 'LSTM':
        model = LSTM(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, output_dim=args.output_dim).to(device)
    elif args.model == 'GRU':
        model = GRU(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, output_dim=args.output_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Choose model and save weights
    checkpoint_path = f'/root/autodl-tmp/stock_prediction/checkpoints/{args.model}_weights.pth'

    # Load weights
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    # Loss function and optimizer
    criterion = torch.nn.MSELoss(reduction='mean').to(device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    hist = np.zeros(args.epochs)
    start_time = time.time()
    lstm = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        # 每5个epoch保存一次权重
        if epoch % 5 == 0:
            torch.save(model.state_dict(), checkpoint_path)

        hist[epoch - 1] = loss.item()
        print(f"Epoch {epoch}/{args.epochs} Loss: {loss.item()}")

    training_time = time.time() - start_time
    print(f"Training time: {training_time}")

    # Predictions and evaluation
    model.eval()
    with torch.no_grad():
        y_train_pred = model(x_train.to(device))
        y_test_pred = model(x_test.to(device))

    # Inverse transform and calculate RMSE
    y_train_pred = scaler.inverse_transform(y_train_pred.cpu().numpy())
    y_train = scaler.inverse_transform(y_train.cpu().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.cpu().numpy())
    y_test = scaler.inverse_transform(y_test.cpu().numpy())

    train_score = np.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
    test_score = np.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print(f'Train Score: {train_score:.2f} RMSE')
    print(f'Test Score: {test_score:.2f} RMSE')
    lstm.append(train_score)
    lstm.append(test_score)
    lstm.append(training_time)

    # Plot results
    train_predict_plot = np.empty_like(price[['Close']])
    train_predict_plot[:, :] = np.nan
    train_predict_plot[lookback:len(y_train_pred) + lookback, :] = y_train_pred

    test_predict_plot = np.empty_like(price[['Close']])
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(y_train_pred) + lookback - 1:len(price[['Close']]) - 1, :] = y_test_pred

    original = scaler.inverse_transform(price[['Close']].values.reshape(-1, 1))
    predictions = np.append(train_predict_plot, test_predict_plot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result[0], mode='lines', name='Train prediction'))
    fig.add_trace(go.Scatter(x=result.index, y=result[1], mode='lines', name='Test prediction'))
    fig.add_trace(go.Scatter(x=result.index, y=result[2], mode='lines', name='Actual Value'))
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template='plotly_dark'
    )
    fig.update_layout(annotations=[dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                        xanchor='left', yanchor='bottom',
                                        text='Results',
                                        font=dict(family='Rockwell',
                                                  size=26,
                                                  color='white'),
                                        showarrow=False)])
    fig.show()

if __name__ == '__main__':
    main()
