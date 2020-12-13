import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

# ANN 모델 생성
class Model(nn.Module):
    def __init__(self, in_features, h1=8,h2=9, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, out_features)

    #순전파
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
# End of class Model:

class AnnModel:
    def __init__(self, x_train, y_train):
        in_features = x_train.dim()
        self.model = Model(x_train.size()[1])
        #self.model = Model(1)
        #손실함수 정의
        self.loss_f = torch.nn.CrossEntropyLoss()
        #최적화 함수 정의
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        dataset = TensorDataset(x_train, y_train)
        self.dataloader = DataLoader(dataset, batch_size=1000)

    def train(self):
        for batch_idx, samples in enumerate(self.dataloader):
            x_train, y_train = samples
            prediction = self.model(x_train)

            cost = self.loss_f(prediction, y_train)
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()
            print(f'Batch: {batch_idx + 1}/{len(self.dataloader)} cost: {cost.item()}')

    def test(self, x_test, y_test):
        correct0 = 0
        correct1 = 0
        no = 0
        with torch.no_grad():
            for i, data in enumerate(x_test):
                y_val = self.model.forward(data)
                print(f'{i+1}. => {str(y_val.argmax().item())} {y_test[i]}')
                if y_val.argmax().item() == y_test[i]:
                    if y_test[i]:
                        correct1 += 1
                    else:
                        correct0 += 1
                else:
                    no += 1
        print(f'We got {correct0}, {correct1} correct no {no} : {(correct0+correct1)/(correct0+correct1+no)}.')

# End of class AnnModel: