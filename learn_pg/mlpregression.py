import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from learn_pg.bofw import bogrelevance

relevances =  bogrelevance()
# Assuming you have your data in numpy arrays X and y
X = [x['vec'] for x in relevances]  # Placeholder
y = [x['score'] for x in relevances]  # Placeholder

# Normalize the data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train , dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test , dtype=torch.float32).view(-1, 1)


# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(7940, 4000)
        self.hidden2 = nn.Linear(4000, 2500)
        self.hidden3 = nn.Linear(2500, 1250)
        self.hidden4 = nn.Linear(1250, 625)
        self.output = nn.Linear(625, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        #x = self.dropout(x)
        x = torch.relu(self.hidden2(x))
        #x = self.dropout(x)
        x = torch.relu(self.hidden3(x))
        #x = self.dropout(x)
        x = torch.relu(self.hidden4(x))
        #x = self.dropout(x)
        x = self.output(x)
        return x


model = MLP()
criterion = nn.MSELoss()  # or nn.L1Loss() for MAE
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
            print(f'Epoch {epoch}, Loss: {loss.item()}, Test Loss: {test_loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()
    mse = mean_squared_error(y_test.numpy(), y_pred)
    mae = mean_absolute_error(y_test.numpy(), y_pred)
    r2 = r2_score(y_test.numpy(), y_pred)
    rmse = np.sqrt(mse)

print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}, RMSE: {rmse}')
