import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


os.makedirs("outputs/HW12b", exist_ok=True)

data_path = "./HOMEWORK1/Housing.csv"
df = pd.read_csv(data_path)

categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype('category').cat.codes)

X = df.drop(columns=["price"]).values
y = df["price"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

class ImprovedHousingMLP(nn.Module):
    def __init__(self, input_size, hidden_dims=[512, 256, 128], dropout_rate=0.1):
        super(ImprovedHousingMLP, self).__init__()
        layers = []
        prev_dim = input_size
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))  
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = ImprovedHousingMLP(X_train.shape[1]).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
model.apply(init_weights)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

early_stopping_patience = 10
best_val_loss = float('inf')
early_stop_counter = 0

EPOCHS = 100
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
 
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    train_loss = criterion(predictions, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor)
        val_loss = criterion(val_predictions, y_test_tensor)
        val_losses.append(val_loss.item())

    scheduler.step(val_loss)  

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
       
        torch.save(model.state_dict(), "outputs/HW12b/best_model.pth")
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {train_loss.item():.4f} - Validation Loss: {val_loss.item():.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss (Improved Housing MLP)")
plt.legend()
plt.savefig("outputs/HW12b/training_validation_loss_improved.png")
plt.show()

model.load_state_dict(torch.load("outputs/HW12b/best_model.pth"))
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).cpu().numpy()
    test_predictions = (test_predictions * y_std) + y_mean  
    y_test_actual = (y_test_tensor.cpu().numpy() * y_std) + y_mean  

    mse = ((test_predictions - y_test_actual) ** 2).mean()
    mae = mean_absolute_error(y_test_actual, test_predictions)
    r2 = r2_score(y_test_actual, test_predictions)

    print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}")
    print(f"Mean Absolute Error (MAE) on Test Set: {mae:.4f}")
    print(f"R² Score on Test Set: {r2:.4f}")

np.savetxt("outputs/HW12b/test_predictions.csv", test_predictions, delimiter=",", header="Predicted Prices", comments="")
np.savetxt("outputs/HW12b/test_ground_truth.csv", y_test_actual, delimiter=",", header="Actual Prices", comments="")
