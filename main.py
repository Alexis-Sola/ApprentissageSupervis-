import torch, torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


corresp_acide = {
    'A': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20,
    '<': '<',
    '<end>': '<'
}

corresp_categorie = {
    '_': 1,
    'h': 2,
    'e': 3,
    '>': '>',
    'nan': '>'
}

#Importer les données
data_train = pd.read_csv("~/Documents/ApplicationIA/protein-secondary-structure-train.txt", delim_whitespace=True)
data_test = pd.read_csv("~/Documents/ApplicationIA/protein-secondary-structure-test.txt", delim_whitespace=True)

#Rename des colonnes
data_train.columns = ["Acide", "Categorie"]
data_test.columns = ["Acide", "Categorie"]



#On supprime les end, on se basera uniquement avec les chevrons
data_train = data_train[data_train.Acide != 'end']
data_train = data_train[data_train.Acide != '<end>']
data_train = data_train.append({'Acide' : '<',
                    'Categorie' : '>'},
                    ignore_index=True)

data_test = data_test[data_test.Acide != 'end']
data_test = data_test[data_test.Acide != '<end>']
data_test = data_test.append({'Acide' : '<',
                    'Categorie' : '>'},
                    ignore_index=True)
print("Info: Import des données terminé")

def ToInt(dataframe):
  new_dataframe = dataframe
  cpt = 0
  for val in dataframe['Acide']:
    new_dataframe['Acide'][cpt] = corresp_acide[val]
    new_dataframe['Categorie'][cpt] = corresp_categorie[str(dataframe['Categorie'][cpt])]
    cpt = cpt + 1
  return new_dataframe

#Conversion des acides en int
data_train = ToInt(data_train)
data_test = ToInt(data_test)
print("Info: conversion en int terminé")

# Définition des features
def ToList(dataframe):
    x_feature = []
    y_cible = []
    tmp = [0, 0]
    tmp2 = [0, 0]
    for i in range(len(dataframe)):
        if i > 0:
            mem = dataframe.loc[i - 1, "Acide"]

        acide = dataframe.loc[i, "Acide"]
        categorie = dataframe.loc[i, "Categorie"]
        if acide != '<' and categorie != '>':
            tmp.append(acide)
            tmp2.append(categorie)
        else:
            if mem != acide:
                tmp = tmp + [0, 0]
                tmp2 = tmp2 + [0, 0]
                x_feature.append(tmp)
                y_cible.append(tmp2)
                tmp = [0, 0]
                tmp2 = [0, 0]
    return x_feature, y_cible


def DefineSameSize(m):
    max = 502
    # for tab in m:
    #     if len(tab) > max:
    #         max = len(tab)

    for tab in m:
        if len(tab) < max:
            diff = max - len(tab)
            for i in range(diff):
                tab.append(0)
    return m


train_feature, train_target = ToList(data_train)
test_feature, test_target = ToList(data_test)
print("Info: Split list terminé")

train_feature = DefineSameSize(train_feature)
train_target = DefineSameSize(train_target)
test_feature = DefineSameSize(test_feature)
test_target = DefineSameSize(test_target)
print("Info: 0 padding terminé")


def OneHot(nbToConvert, nbClasse):
  tabOneHot = np.zeros(nbClasse)
  if nbToConvert == 0:
    return tabOneHot

  tabOneHot[nbToConvert - 1] = 1
  return tabOneHot


def ConvertAllTab(array2d, nbClasse):
  tmp = []
  for tab in array2d:
    tab_current = []
    for val in tab:
      tab_current.append(OneHot(val, nbClasse))
    tmp.append(tab_current)
  return np.asarray(tmp)


x_train = torch.tensor(ConvertAllTab(train_feature, 20))
y_train = torch.tensor(ConvertAllTab(train_target, 3))
x_test = torch.tensor(ConvertAllTab(test_feature, 20))
y_test = torch.tensor(ConvertAllTab(test_target, 3))

# x_train = F.one_hot(torch.from_numpy(np.asarray(train_feature)))
# x_test = F.one_hot(torch.from_numpy(np.asarray(test_feature)))
# y_train = F.one_hot(torch.from_numpy(np.asarray(train_target)))
# y_test = F.one_hot(torch.from_numpy(np.asarray(test_target)))
print("Info: Conversion en hot terminé")


#Définition du model
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(self.hidden_size, 3)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        output = self.softmax(x)
        return output



model = MLP(20, 300)
criterion = torch.nn.BCELoss()
#criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)



x_train_final = x_train.detach().clone().float()
y_train_final = y_train.detach().clone().float()
x_test_final = x_test.detach().clone().float()
y_test_final = y_test.detach().clone().float()

#Vérifie si l'on est sur une valeur du 0 padding
def Mask(tab):
    for val in tab:
        if val == 1.:
            return False
    return True

#Permet de calculer l'accuracy de notre model
def IsGoodAnswer(tab1, tab2):
    val1 = torch.argmax(tab1)
    val2 = torch.argmax(tab2)
    if val1 != val2:
        return False
    return True


epoch = 20
for epoch in range(epoch):
    running_loss = 0.0
    cpt = 0
    predictionEpoch = 0
    best = 1
    #On parcours nos séquences d'acides
    for i, data in enumerate(x_train_final, 0):
        loss_acide = 0
        cpt = cpt + 1
        cpt2 = 0
        goodPrediction = 0
        #on parcours chaque acide
        for j, acide in enumerate(data, 0):
            cpt2 += 1
            #Si l'on est sur une valeur du 0 padding
            #On ne prend pas en compte pour l'apprentissage
            if Mask(np.asarray(acide)):
                continue

            model.train()
            #Nettoyage du gradient
            optimizer.zero_grad()

            #Prédiction faites par le model
            outputs = model(acide)

            #On récupère le pourcentage de bonne réponse
            answer = y_train_final[i][j]
            if IsGoodAnswer(outputs, answer):
                goodPrediction += 1

            #On calcul le loss
            loss_train = criterion(outputs, answer)
            #Mise à jour des poids
            loss_train.backward()
            optimizer.step()
            loss_acide += loss_train.item()

        #Calcul du loss et de l'accuracy
        running_loss += loss_acide/cpt2
        if running_loss < best:
            best = running_loss
        else:
            break
        predictionEpoch = goodPrediction/cpt2

    print('Epoch :', epoch + 1, 'Loss :', running_loss/cpt, "Accuracy :", predictionEpoch)


#On évalue le model sur le jeu de test
model.eval()
y_pred = model(x_test_final)
loss = criterion(y_pred, y_test_final)
print('Test loss', loss.item())
