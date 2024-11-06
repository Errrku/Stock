import argparse

def args_parse():
    parser = argparse.ArgumentParser(description="Stock Price Prediction")
    
    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'GRU'], help='Model type: LSTM or GRU')

    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--lookback', type=int, default=20, help="Sequence length for LSTM")
    parser.add_argument('--hidden_dim', type=int, default=32, help="Number of hidden units in LSTM")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers")
    parser.add_argument('--input_dim', type=int, default=1, help="Input dimension")
    parser.add_argument('--output_dim', type=int, default=1, help="Output dimension")
    
    return parser.parse_args()
