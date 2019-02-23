import pandas as pd
import numpy as np
import torch
import torchvision
import os
from PIL import Image
import PIL
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

# Установим размер классифицируемых изображений
PIC_SIZE = 50
# Путь к предобработанным данным
data_path = 'data//preprocessed//'
# Путь, куда сохраним модель
model_save_path = 'signs_classifier.pth'


class SignsDataset(Dataset):
    """Road signs dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.signs_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Cоздаём массив label->index и массив index->label
        self.labels = self.signs_frame['label'].unique()
        self.label_indexes = {}
        for i, label in enumerate(self.labels):
            self.label_indexes[label] = i

    def __len__(self):
        return len(self.signs_frame)

    def __getitem__(self, idx):
        # Загрузим изображение и приведём к размеру 50х50
        img_name = self.root_dir + self.signs_frame.iloc[idx, 0]
        image = Image.open(img_name)
        image = image.resize((PIC_SIZE, PIC_SIZE), Image.ANTIALIAS)

        # В роли ответа будем давать номер label
        label_string = self.signs_frame.iloc[idx, 1]
        label = self.label_indexes[label_string]

        # Применим преобразования изображения (например аугментацию)
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample

dataset = SignsDataset(data_path + 'labels.csv',
                       data_path,
                       torchvision.transforms.ToTensor())

indicies = np.arange(len(dataset))


# Разбиение датасета на train и validation
train_sampler = SubsetRandomSampler(indicies[:int(len(dataset)*0.5)])
validation_sampler = SubsetRandomSampler(indicies[int(len(dataset)*0.5):])

# DataLoader достаёт данные из dataset батчами
signsTrainLoader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
signsValidationLoader = DataLoader(dataset, batch_size=32, sampler=validation_sampler)


df = dataset.signs_frame
classes_number = df['label'].nunique()
print('Classes number:', classes_number)
df.groupby('label')['file_name'].nunique()

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
batch = next(iter(signsTrainLoader))
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
optimizer = torch.optim.Adam(cn.parameters(), lr=learning_rate)

#цикл обучения
i=0

for epoch in tqdm_notebook(range(100)):

    running_loss = 0.0
    for batch in signsTrainLoader:
        #current batch

        X_batch, y_batch = batch['image'], batch['label']

        #обнуляем веса
        optimizer.zero_grad()

        y_pred = cn(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        #пишем в лог каждые 50 батчей
        if i%50 == 49:
            batch = next(iter(signsValidationLoader))
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

batch = next(iter(signsValidationLoader))
predictions = cn(batch['image'])
y_test = batch['label']


#print(predictions, y_test)
_, predictions = torch.max(predictions, 1)
plt.imshow(PIL.ToPIL(batch['image'][0]))
print('Gound-true:', dataset.labels[batch['label'][0]])
print('Prediction:', dataset.labels[predictions[0]])