import torch
import torch.nn as nn
import torch.optim as optim
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RegressionNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


def prepare_data(X, y, batch_size=32, test_size=0.2):
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_normalized = X_scaler.fit_transform(X)
    y_normalized = y_scaler.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_normalized,
        test_size=test_size,
        random_state=42
    )

    train_dataset = RegressionDataset(X_train, y_train)
    test_dataset = RegressionDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, (X_scaler, y_scaler)


def train_model(model, train_loader, criterion, optimizer, device, epochs=100):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    return train_losses


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    return avg_loss, np.array(predictions), np.array(actuals)


def plot_results(train_losses, predictions, actuals, y_scaler=None):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    if y_scaler is not None:
        predictions = y_scaler.inverse_transform(predictions)
        actuals = y_scaler.inverse_transform(actuals)

    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
    plt.title('Predictions vs Actuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend('best')
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = pd.read_csv('ChinaBank.csv')

    X = data.iloc[1:, 2:5].values  # Added .values to ensure numpy array
    y = data.iloc[1:, 5:6].values

    train_loader, test_loader, scalers = prepare_data(X, y, batch_size=32)
    X_scaler, y_scaler = scalers

    model = RegressionNet(input_dim=3, hidden_dim=32).to(device)  # Changed input_dim to 3
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epochs=100
    )

    test_loss, predictions, actuals = evaluate_model(
        model,
        test_loader,
        criterion,
        device
    )
    print(f'Test Loss: {test_loss:.4f}')

    plot_results(train_losses, predictions, actuals, y_scaler)

    return model, scalers


if __name__ == '__main__':
    model, scalers = main()