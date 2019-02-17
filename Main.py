import pandas as pd
import numpy as np
import torch
import torchvision
import os
from PIL import Image
import torch.vision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

num_epochs = 5
num_classes = 10
batch_size = 16
learning_rate = 0.001

PIC_SIZE = 50
DATA_PATH = ''
MODEL_STORE_PATH = 'C:\\Users\Andy\PycharmProjects\pytorch_models\\'

class PersonDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.persons_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.persons = self.persons_frame['person'].unique()
        self.person_indexes = {}
        for i, person in enumerate(self.persons):
            self.person_indexes[person] = i
    def __len__(self):
        return len(self.persons_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.persons_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.resize((PIC_SIZE, PIC_SIZE), Image.ANTIALIAS)
        image = np.array(image)

        person_string = self.persons_frame.iloc[idx, 1]
        person = self.person_indexes[person_string]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'person': person}

        return sample

dataset = PersonDataset(DATA_PATH + 'persons.csv', DATA_PATH, torchvision.transforms.ToTensor())

indicies = np.arange(len(dataset))

train_sampler = SubsetRandomSampler(indicies[:int(len(dataset)*0.5)])
validation_sampler = SubsetRandomSampler(indicies[int(len(dataset)*0.5):])

personsTrainLoader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
personsValidationLoader = DataLoader(dataset, batch_size=32, sampler=validation_sampler)

ToPIL = transforms.ToPILImage()

batch = next(iter(personsTrainLoader))

img = batch['image'][0]
label_index = batch['person'][0]

print(dataset.labels[label_index])
plt.inshow(ToPIL(img))

df = dataset.signs_frame

#СОЗДАЕМ СЕТЬ
#
#
#
#
#

import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, class_number):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, 5)

        self.fc1 = nn.Linear(5776, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_number)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))

        x = x.view((-1, 5776))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cn = ConvNet(num_classes)
batch = next(iter(personsTrainLoader))
cn(batch['image'])[0]


#ОБУЧАЕМ
#
#
#
#
#


history = {'loss':[], 'val_loss':[]}

#Функция потерь
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-4
optimizer = torch.optim.Adam(cn.parameters(), Ir = learning_rate)

#цикл обучения
i=0

for epoch in tqdm_notebook(range(100)):

    running_loss = 0.0
    for batch in personsTrainLoader:
        #current batch

        X_batch, y_batch = batch['image'], batch['label']

        #обнуляем веса
        optimizer.zero_grad()

        y_pred = cn(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        ###########
        running_loss += loss.item()
        #пишем в лог каждые 50 батчей
        if i%50 == 49:
            batch = next(iter(personsValidationLoader))
            X_batch, y_batch = batch['image'], batch['label']
            y_pred = cn(X_batch)

            history['loss'].append(loss.item())
            history['val_loss'].append(loss_fn(y_pred, y_batch).item())

        #качество
        if i%1000 == 999:
            print('[%d, %Sd] loss: %.3f' % (epoch + 1, i+1, running_loss/1000))
            running_loss = 0
        i += 1


    #saving model
    torch.save(cn.state_dict(), model_save_path)
    print('обучение закончено')


cn = ConvNet(num_classes)
cn.load_state_dict(torch.load(model_save_path))