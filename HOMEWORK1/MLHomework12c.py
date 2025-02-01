import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

os.makedirs("outputs/HW12c", exist_ok=True)

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

class ComplexHousingMLP(nn.Module):
    def __init__(self, input_size, hidden_dims=[1024, 512, 256, 128, 64, 32], dropout_rate=0.1):
        super(ComplexHousingMLP, self).__init__()
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

complex_model = ComplexHousingMLP(X_train.shape[1]).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
complex_model.apply(init_weights)

criterion = nn.MSELoss()
optimizer = optim.Adam(complex_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

EPOCHS = 100
complex_train_losses = []
complex_val_losses = []

for epoch in range(EPOCHS):
   
    complex_model.train()
    optimizer.zero_grad()
    predictions = complex_model(X_train_tensor)
    train_loss = criterion(predictions, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    complex_train_losses.append(train_loss.item())

    
    complex_model.eval()
    with torch.no_grad():
        val_predictions = complex_model(X_test_tensor)
        val_loss = criterion(val_predictions, y_test_tensor)
        complex_val_losses.append(val_loss.item())

    scheduler.step(val_loss)  

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {train_loss.item():.4f} - Validation Loss: {val_loss.item():.4f}")

torch.save(complex_model.state_dict(), "outputs/HW12c/complex_model.pth")
print("Complex model saved to 'outputs/HW12c/complex_model.pth'")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(complex_train_losses) + 1), complex_train_losses, label="Complex Model Training Loss")
plt.plot(range(1, len(complex_val_losses) + 1), complex_val_losses, label="Complex Model Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss (Complex Model)")
plt.legend()
plt.savefig("outputs/HW12c/complex_model_loss.png")
plt.show()

complex_model.eval()
with torch.no_grad():
    complex_test_predictions = complex_model(X_test_tensor).cpu().numpy()
    complex_test_predictions = (complex_test_predictions * y_std) + y_mean  
    y_test_actual = (y_test_tensor.cpu().numpy() * y_std) + y_mean  

    complex_mse = ((complex_test_predictions - y_test_actual) ** 2).mean()
    complex_r2 = r2_score(y_test_actual, complex_test_predictions)

    print(f"Complex Model - Mean Squared Error (MSE): {complex_mse:.4f}")
    print(f"Complex Model - R² Score: {complex_r2:.4f}")

np.savetxt("outputs/HW12c/complex_test_predictions.csv", complex_test_predictions, delimiter=",", header="Predicted Prices", comments="")
np.savetxt("outputs/HW12c/complex_test_ground_truth.csv", y_test_actual, delimiter=",", header="Actual Prices", comments="")
print("Predictions saved to 'outputs/HW12c/'")
