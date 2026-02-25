from load_and_preprocess import *
import joblib
from load_data import load_data
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_new, original_df, healthy_df, scaler = load_and_preprocess('train_FD001.txt')
#joblib.dump(scaler, 'scaler.pkl')
RUL_MAX = 125
batch_size=32
def compute_HI(predictions, RUL_MAX):
    predictions = np.clip(predictions, 0, RUL_MAX)
    return predictions/RUL_MAX

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

train_loader, test_loader, test_engine_list, test_cycle_list = load_data(
    df=df_new,
    window_size=30,
    columns=sensor_cols,
    stride=1,
    test_size=0.2,
    batch_size=batch_size
)

#print(len(train_loader), len(test_loader))


def plot_loss_curves(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, label="Training Loss", color='blue')
    plt.plot(val_loss, label="Validation Loss", color='red')
    plt.savefig("Loss_curves.png")
    plt.close()


def plot_RUL(all_engines, all_cycles, all_preds, all_trues, all_hi):
    results_df = pd.DataFrame({
        "engines": all_engines,
        "cycles": all_cycles,
        "true RUL": all_trues,
        "pred RUL": all_preds,
        "HI" : all_hi
    })
    results_df.to_csv('results_df.csv', index=False)
    results_df = results_df.sort_values(["engines", "cycles"])
    engine_id = results_df["engines"].unique()[0]
    engine_df = results_df[results_df["engines"] == engine_id]

    plt.figure()
    plt.plot(engine_df["cycles"], engine_df["true RUL"], label="True RUL")
    plt.plot(engine_df["cycles"], engine_df["pred RUL"], label="Prediction")
    plt.legend()
    plt.xlabel("Cycles")
    plt.ylabel("RUL")
    plt.savefig(f"RUL_comparison_{engine_id}.png")
    plt.close()

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

num_epochs=50
model = LSTM_model(input_size=21, hidden_size=64, output_size=1)

#input_size: number of sensors
# output_size :  one number prediction, RUL

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss=[]
validation_loss=[]
rmse_train=[]
rmse_validation=[]
all_hi = []
all_preds = []
all_trues= []
all_engines = []
all_cycles = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_rmse_train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.float().unsqueeze(1)
        data=data.float()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        rmse_loss_batch = torch.sqrt(torch.mean((outputs - target)**2))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_rmse_train_loss += rmse_loss_batch.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx+1}, RMSE Loss {rmse_loss_batch.item()}, Training Loss {loss.item()}")

    epoch_loss = running_loss / len(train_loader)
    epoch_rmse_loss = running_rmse_train_loss / len(train_loader)
    train_loss.append(epoch_loss)
    rmse_train.append(epoch_rmse_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss {epoch_loss}")

    model.eval()
    running_validation_loss=0.0
    running_rmse_val_loss=0.0
    idx=0
    with torch.no_grad():
        for data_test, target_test in test_loader:
            target_test = target_test.float().unsqueeze(1)
            true = target_test.squeeze().numpy()

            test_data = data_test.float()

            outputs_test = model(data_test)
            pred = outputs_test.squeeze().numpy()

            val_loss = criterion(outputs_test, target_test)
            rmse_loss_validation = torch.sqrt(torch.mean((outputs_test - target_test)**2))
            running_validation_loss += val_loss.item()
            running_rmse_val_loss += rmse_loss_validation.item()
            #plot_predictions(target_test, outputs_test, 2, epoch)

            HI_val = compute_HI(outputs_test, RUL_MAX)
            all_hi.extend(HI_val.tolist())

            all_engines.extend(test_engine_list[idx:idx+batch_size])
            all_cycles.extend(test_cycle_list[idx:idx+batch_size])
            all_preds.extend(pred.tolist())
            all_trues.extend(true.tolist())
            idx+=batch_size

        epoch_val_loss = running_validation_loss / len(test_loader)
        epoch_rmse_val_loss = running_rmse_val_loss / len(test_loader)
        validation_loss.append(epoch_val_loss)
        rmse_validation.append(epoch_rmse_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation RMSE loss {epoch_rmse_val_loss}, Validation Loss {epoch_val_loss}")
torch.save(model, "model.pth")
joblib.dump(scaler, 'scaler.pkl')

plot_loss_curves(np.array(train_loss), np.array(validation_loss))
plot_RUL(all_engines, all_cycles, all_preds, all_trues, all_hi)

all_hi = np.array(all_hi)
print("HI statistics:")
print(f"Mean: {all_hi.mean():.3f}")
print(f"Min: {all_hi.min():.3f}")
print(f"Max: {all_hi.max():.3f}")
print(f"Std: {all_hi.std():.3f}")

# Optional: visualize
plt.hist(all_hi, bins=30, color='skyblue')
plt.xlabel("Health Index")
plt.ylabel("Frequency")
plt.title("Distribution of Health Index across evaluation data")
plt.savefig("HI_distribution.png")
plt.close()




