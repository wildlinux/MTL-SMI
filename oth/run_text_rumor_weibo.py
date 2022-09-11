import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from TEXT_RUMOR_WEIBO import TEXT_RUMOR_WEIBO
from utils_text import load_data_weibo

X_train_tid, X_dev_tid, X_test_tid, X_train, X_dev, X_test, y_train, y_dev, y_test = load_data_weibo()

config = {'maxlen': 50, 'dropout': 0.5, 'n_heads': 8, 'kernel_sizes': [5, 6, 7], 'num_classes': 2, 'epochs': 3,
          'batch_size': 64, 'learning_rate': 0.0005}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TEXT_RUMOR_WEIBO(config)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

X_train_tid = torch.LongTensor(X_train_tid)
X_train = torch.LongTensor(X_train)
y_train = torch.LongTensor(y_train)

dataset = TensorDataset(X_train_tid, X_train, y_train)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
loss = nn.CrossEntropyLoss()
# none表示不对损失项reduce, 也就是不对各个损失项进行求和取平均
criterion = nn.CrossEntropyLoss(reduction='none')
for epoch in range(config['epochs']):
    print("\nEpoch ", epoch + 1, "/", config['epochs'])
    model.train()
    avg_loss = 0
    avg_acc = 0
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        total = len(dataloader)
        print(i,total)
        with torch.no_grad():
            batch_x_tid, batch_x_text, batch_y = (item.cuda(device=device) for item in data)
        output = model.forward(batch_x_text)
        loss_value = loss(output, batch_y)
        loss_value.backward()
        optimizer.step()
