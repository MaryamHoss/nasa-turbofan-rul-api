import torch
import torch.nn as nn
import pandas as pd

def transform_window(scaler, window, sensor_cols):
    #Scale and convert to tensor
    df_window = pd.DataFrame(window, columns=sensor_cols)
    transformed_window = scaler.transform(df_window)
    transformed_window = torch.tensor(transformed_window.values, dtype=torch.float32)
    transformed_window = transformed_window.unsqueeze(0)
    return transformed_window

def predict_output(model, window):

    with torch.no_grad():
        output = model(window)
        predicted_RUL = output.item()
        return predicted_RUL


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #x has shape: batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)   # lstm_out: (batch, seq_len, hidden_size)
        last_out = lstm_out[:, -1, :]  # take output at last time step
        output = self.dropout(last_out)
        output = self.fc(output)

        return output  # out: (batch_size, output_size)

