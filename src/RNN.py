import torch
from torch.utils.data import dataloader
import torch.nn as nn
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


class RnnTrainer:
    def __init__(self, model, loss_fn, optimizer,n_history):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.n_history = n_history
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def train_step(self, x, y):
        self.model.train()
        preds = self.model(x)
        loss = self.loss_fn(y, preds.to(self.device))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features=1):
        # model_path = f'models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        t = trange(1, n_epochs + 1)
        for epoch in t:
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, 3, self.n_history]).to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    preds = self.model(x_val)
                    val_loss = self.loss_fn(y_val, preds.to(self.device)).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            
            t.set_description(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")
            t.refresh() # to show immediately the update


        # torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, 3, self.n_history]).to(self.device)
                y_test = y_test.to(self.device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def inverse_transform(self,scaler, df, columns):
        for col in columns:
            df[col] = scaler.inverse_transform(df[col])
        return df


    def format_predictions(self,test_loader,scaler):
        predictions,values = self.evaluate(test_loader)
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds})
        df_result = df_result.sort_index()
        df_result = self.inverse_transform(scaler, df_result, [["value", "prediction"]])
        return df_result

    def score(self,test_loader,scaler):
        scorable_df = self.format_predictions(test_loader,scaler)
        return {'mae' : mean_absolute_error(scorable_df.value, scorable_df.prediction),
            'mape ' : mean_absolute_percentage_error(scorable_df.value,scorable_df.prediction),
            'rmse' : mean_squared_error(scorable_df.value, scorable_df.prediction) ** 0.5,
            'r2' : r2_score(scorable_df.value, scorable_df.prediction)}





class VanillaRNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=1, num_layers=1, output_dim=1, dropout_prob=.3):
        super(VanillaRNN, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, h0 = self.rnn(x.cpu(), h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x.cpu(), (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)

        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, _ = self.gru(x.cpu(), h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)

        return out


