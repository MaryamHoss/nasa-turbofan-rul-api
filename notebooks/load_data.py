from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

def add_RUL(df):
    df['RUL']= df.groupby('units')['cycles'].transform('max') - df['cycles']
    df['RUL'] = df['RUL'].clip(upper=125)
    return df

def load_data(df, window_size, columns, stride, test_size, batch_size):
    df = add_RUL(df).copy()

    # Split engines
    engine_ids = df['units'].unique()
    train_ids, test_ids = train_test_split(engine_ids, test_size=test_size, random_state=42)


    # Lists to hold windows and labels
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    test_engine_lists = []
    test_sycles_list = []

    # Generate windows
    for u_id, groups in df.groupby('units'):
        for x in range(0, len(groups) - window_size + 1, stride):

            window = groups[columns].iloc[x:x + window_size].values
            label = groups['RUL'].iloc[x + window_size - 1]

            if u_id in train_ids:
                X_train_list.append(window)
                y_train_list.append(label)
            elif u_id in test_ids:
                X_test_list.append(window)
                y_test_list.append(label)
                test_engine_lists.append(u_id)
                test_sycles_list.append(groups['cycles'].iloc[x + window_size - 1])

    # Convert to tensors
    X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_test_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test_list), dtype=torch.float32)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_engine_lists, test_sycles_list