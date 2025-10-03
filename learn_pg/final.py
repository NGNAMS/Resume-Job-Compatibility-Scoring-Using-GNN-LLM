import torch
from learn_pg.model import SimpleGCN
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import numpy as np



data_set = torch.load('small_skill_dataset_six_point_five.pt')

train_dataset, test_dataset = train_test_split(data_set, test_size=0.2, random_state=42)
model = SimpleGCN(1536, 1024)

train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

num_epochs = 200  # Number of training epochs

for epoch in range(num_epochs + 1):
    model.train()  # Set the model to training mode
    total_loss = 0

    for data in train_loader:
        outputs = model(data)
        targets = data.y / 100
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    #print(f'Epoch [{epoch+1}/{num_epochs}], Train_Loss: {avg_loss:.4f}')

    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []


    with torch.inference_mode():
        for data in test_loader:
            outputs = model(data)
            targets = data.y / 100
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    # Flatten the lists of predictions and targets for metric calculations
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Calculate additional metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    medae = median_absolute_error(all_targets, all_predictions)

    if epoch % 20 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, '
              f'RMSE: {rmse:.4f}, MedAE: {medae:.4f}')
