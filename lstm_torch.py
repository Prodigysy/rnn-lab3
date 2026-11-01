'''
        -------   LSTM   -----

Modified from
https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=CKEzO1jzKydL

Lab 3 additions (see методичка, п. 3.2):
  * training time measurement;
  * MSE history saved to CSV while training;
  * loss curve and prediction plot saved to PNG/PDF (a per-run PDF report too);
  * train/test MSE metrics printed;
  * command-line arguments for dataset/hyperparameters.

Usage:
    python lstm_torch.py                                  # airline-passengers.csv
    python lstm_torch.py --dataset data/shampoo.csv --tag shampoo
    python lstm_torch.py --epochs 500 --no-show
'''


import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: we save figures instead of showing them
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# MODEL
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)  # nn.RNN

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def load_data(dataset_name, dataset_label, seq_length=4, out_dir='results', tag=None):
    training_set = pd.read_csv(dataset_name)
    training_set = training_set.iloc[:, 1:2].values

    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    x, y = sliding_windows(training_data, seq_length)

    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    dataX = torch.Tensor(np.array(x))
    dataY = torch.Tensor(np.array(y))

    trainX = torch.Tensor(np.array(x[0:train_size]))
    trainY = torch.Tensor(np.array(y[0:train_size]))

    testX = torch.Tensor(np.array(x[train_size:len(x)]))
    testY = torch.Tensor(np.array(y[train_size:len(y)]))

    return (sc, train_size, test_size, dataX, dataY,
            trainX, trainY, testX, testY, training_set)


def plot_series(series, label, path_png, path_pdf=None):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(series, color='tab:blue')
    ax.set_title(label)
    ax.set_xlabel('time step')
    ax.set_ylabel('value')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    if path_pdf:
        fig.savefig(path_pdf)
    plt.close(fig)


# TRAIN and TEST

def train_and_test_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    seq_length = args.seq_length
    num_epochs = args.epochs
    learning_rate = args.lr
    input_size = 1
    hidden_size = args.hidden
    num_layers = args.layers
    num_classes = 1

    tag = args.tag or os.path.splitext(os.path.basename(args.dataset))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    (sc, train_size, test_size, dataX, dataY,
     trainX, trainY, testX, testY, training_set) = load_data(
        args.dataset, args.label, seq_length=seq_length,
        out_dir=args.out_dir, tag=tag)

    series_png = os.path.join(args.out_dir, tag + '_series.png')
    plot_series(training_set, args.label, series_png)

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model
    history = []
    t0 = time.perf_counter()

    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, trainY)

        loss.backward()

        optimizer.step()
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            history.append((epoch, loss.item()))
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    elapsed = time.perf_counter() - t0

    lstm.eval()
    with torch.no_grad():
        all_predict = lstm(dataX)

    data_predict = all_predict.numpy()
    dataY_plot = dataY.numpy()

    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)

    # Metrics: MSE in scaled units and RMSE in the original units.
    train_mse = float(np.mean((data_predict[:train_size] - dataY_plot[:train_size]) ** 2))
    test_mse = float(np.mean((data_predict[train_size:] - dataY_plot[train_size:]) ** 2))
    train_rmse = float(np.sqrt(np.mean((data_predict[:train_size] - dataY_plot[:train_size]) ** 2)))
    test_rmse = float(np.sqrt(np.mean((data_predict[train_size:] - dataY_plot[train_size:]) ** 2)))

    # Save the loss history.
    csv_path = os.path.join(args.out_dir, tag + '_loss.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_mse'])
        w.writerows(history)

    # Prediction plot (same as the original notebook, but saved to files).
    pred_png = os.path.join(args.out_dir, tag + '_prediction.png')
    pred_pdf = os.path.join(args.out_dir, tag + '_prediction.pdf')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvline(x=train_size, c='r', linestyle='--', label='train/test boundary')
    ax.plot(dataY_plot, label='actual')
    ax.plot(data_predict, label='predicted')
    ax.set_xlabel('time step')
    ax.set_ylabel('value')
    ax.set_title('Time-Series Prediction — %s' % args.label)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pred_png, dpi=150)
    fig.savefig(pred_pdf)
    plt.close(fig)

    # Loss curve.
    loss_png = os.path.join(args.out_dir, tag + '_loss.png')
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot([h[0] for h in history], [h[1] for h in history], color='tab:green')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE (scaled)')
    ax.set_title('Training loss — %s' % args.label)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)

    summary = {
        'model': 'nn.LSTM (PyTorch)',
        'dataset': args.dataset,
        'dataset_label': args.label,
        'samples': int(len(dataY)),
        'train_size': int(train_size),
        'test_size': int(test_size),
        'seq_length': seq_length,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'epochs': num_epochs,
        'learning_rate': learning_rate,
        'training_seconds': round(elapsed, 3),
        'final_train_mse_scaled': round(float(history[-1][1]), 6),
        'train_mse_original_units': round(train_mse, 3),
        'test_mse_original_units': round(test_mse, 3),
        'train_rmse_original_units': round(train_rmse, 3),
        'test_rmse_original_units': round(test_rmse, 3),
    }

    summary_path = os.path.join(args.out_dir, tag + '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Per-run PDF report.
    report_path = os.path.join(args.out_dir, tag + '_report.pdf')
    with PdfPages(report_path) as pdf:
        for png in (series_png, loss_png, pred_png):
            img = plt.imread(png)
            fig = plt.figure(figsize=(11, 6))
            ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        fig.text(0.03, 0.95, 'LSTM time-series run — %s' % args.label,
                 fontsize=14, fontweight='bold', va='top')
        y = 0.88
        for k, v in summary.items():
            fig.text(0.03, y, '%s: %s' % (k, v), fontsize=10,
                     family='monospace', va='top')
            y -= 0.035
        pdf.savefig(fig)
        plt.close(fig)

    print('\nTraining time: %.2f s' % elapsed)
    print('RMSE (original units): train %.3f, test %.3f' % (train_rmse, test_rmse))
    print('Saved: %s' % ', '.join([series_png, loss_png, csv_path, pred_png,
                                   pred_pdf, report_path, summary_path]))

    if args.show:
        plt.show()

    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='LSTM time-series prediction (Lab 3)')
    ap.add_argument('--dataset', default=os.path.join('data', 'airline-passengers.csv'))
    ap.add_argument('--label', default='Airline Passengers Data')
    ap.add_argument('--tag', default=None, help='output file prefix')
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--seq-length', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=2)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--show', action='store_true', help='show figures interactively')
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train_and_test_model(args)
