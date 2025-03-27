
import numpy as np
import math
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
p = 0
while p<197 :
 with open("all51.csv") as f:
  array = np.loadtxt(f, delimiter=",")
  num = array[p, :]
  #print(num)
  x=int(np.sum(num))
  #print(x)
  atom = array[p+1:x+p+1, :-3]
  cell = array[x+p+1:x+p+4, :]
  elec = array[p+1:x+p+1, 3]
  affi = array[p+1:x+p+1, 4]
  Ei = array[p+1:x+p+1, 5]
  mas = array[p+1:x+p+1, 6]
  radiu = array[p+1:x+p+1, 7]
  p = p+x+4
  sh=0
  atoms=np.zeros((25*x,3))  # 400 is the max nmber of atoms
  electro=np.zeros((25*x))  # 400 is the max nmber of atoms
  affinity=np.zeros((25*x))
  Eion=np.zeros((25*x))
  mass=np.zeros((25*x))
  radius=np.zeros((25*x))
  for ix in range(-2, 3):
   for iy in range(-2, 3):
    for ik in range(0, 4):
     atoms[sh,0]=atom[ik,0]+ix*cell[0,0]+iy*cell[1,0]
     atoms[sh,1]=atom[ik,1]+ix*cell[0,1]+iy*cell[1,1]
     atoms[sh,2]=atom[ik,2]
     electro[sh]=elec[[ik]]
     affinity[sh]=affi[[ik]]
     Eion[sh]=Ei[[ik]]
     mass[sh]=mas[[ik]]
     radius[sh]=radiu[[ik]]
     sh=sh+1


     r=np.zeros(10)
     teta=np.zeros(10)
     fi=np.zeros(10)
     finger1=np.zeros((25,25,25))
     finger2=np.zeros((25,25,25))
     finger3=np.zeros((25,25,25))
     finger4=np.zeros((25,25,25))
     finger5=np.zeros((25,25,25))

     rmax=(((cell[0,0]+cell[0,1])**2)+((cell[1,0]+cell[1,1])**2))**0.5
     rmax=rmax*5.5
  for ia in range(0, x):
   l=(atoms[ia,0]**2+atoms[ia,1]**2+atoms[ia,2]**2)**0.5
   l=int(l/(rmax/10))
   m=int((math.atan(atoms[ia,2]/((atoms[ia,0]**2+atoms[ia,1]**2)**0.5)))/(math.pi/10))
   n=int((math.acos(atoms[ia,0]/((atoms[ia,0]**2+atoms[ia,1]**2)**0.5)))/((2*math.pi)/10))
   #print(l,m,n)
  finger1[[l,m,n]]=finger1[[l,m,n]]+electro[[ia]]
  finger2[[l,m,n]]=finger2[[l,m,n]]+affinity[[ia]]
  finger3[[l,m,n]]=finger3[[l,m,n]]+Eion[[ia]]
  finger4[[l,m,n]]=finger4[[l,m,n]]+mass[[ia]]
  finger5[[l,m,n]]=finger5[[l,m,n]]+radius[[ia]]
  out = np.concatenate((finger1, finger2, finger3, finger4, finger5))
  #np.savetxt("fingers.csv", finger1, delimiter=",")
  new_out = out.reshape((out.shape[0]*out.shape[1]), out.shape[2])
  new_out = new_out.transpose()
  np.savetxt("inputfingers.csv", new_out, delimiter=",")

from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('inputfingers.csv')
data = pd.read_csv('inputfingers.csv')
#data = dataframe.values
np.savetxt("data.csv", data, delimiter=",")
#data = data.reshape((25, 3125))
with open("target.csv") as f:
 y = np.loadtxt(f, delimiter=",")
np.savetxt("y.csv", y, delimiter=",")
from sklearn.model_selection import train_test_split
data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.1)
np.savetxt("y_train.csv", y_train, delimiter=",")
np.savetxt("y_test.csv", y_train, delimiter=",")
np.savetxt("data_train.csv", data_test, delimiter=",")
np.savetxt("data_test.csv", data_test, delimiter=",")

model = Sequential()
model.add(Dense(64, input_dim=3125, activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1000,activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
model.fit(data, y, epochs=100, batch_size=16)
datatest = model.predict(data_test)
datatrain = model.predict(data_train)
np.savetxt("train_preds.csv", train_preds, delimiter=",")
