import pandas as pd
import torch
import torch.nn as nn
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Data:
    
    def __init__(self, target_col):
        metmast = pd.read_csv('./data/window/metmast.csv', sep=',')
        metmast['Date/time'] = pd.to_datetime(metmast['Date/time'])
        metmast.set_index('Date/time', inplace=True)
        
        turbine = pd.read_csv('./data/window/turbine_data.csv', sep=',')
        turbine['Zeitstempel (UTC)'] = pd.to_datetime(turbine['Zeitstempel (UTC)'])
        turbine = turbine.set_index(turbine['Zeitstempel (UTC)'])
        turbine = turbine.drop('Zeitstempel (UTC)', axis=1)
        
        data = pd.merge(metmast, turbine[[target_col]], left_index=True, right_index=True )
        data = data.dropna()
        self.train = torch.tensor(data.iloc[:,:-1].values, dtype=torch.float32)
        self.target = torch.tensor(data.iloc[:,-1].values, dtype=torch.float32)
    
    def __len__(self):
        return self.train.shape[0]
    
    def __getitem__(self, idx):
        return self.train[idx, :], self.target[idx]

class Model(nn.Module):
    
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 40),
            nn.ReLU(),
            nn.Linear(40, 80),
            nn.Sigmoid(),
            nn.Linear(80, 1)                     
        )
        
    def forward(self, features):
        output = self.base(features)
        return output


# Loading the data
dataset = Data(target_col='ws_2')

train_size = len(dataset)-round(0.20*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=32 , shuffle=True)
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# parameters
num_epochs = 50
learning_rate = 0.01

# model
model = Model(input_size=19)


# loss and optimizer --> cross entropy
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_dataset)

for epoch in range(num_epochs):
    for id, (features, label) in enumerate(train_dataset):

        features = features.to(device)
        label = label.to(device)
        
        # forward
        output = model(features)
        loss = criterion(output, label)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (id+1) % 200 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {id+1}/{n_total_steps}, loss = {loss.item():.4f}')
        
                 
        