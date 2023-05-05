import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm 
from os.path import join
from os import listdir 
import cv2
import main as fn
import joblib

imgN=cv2.imread('C:/Users/personal/Documents/tutorial/proyecto/Test/Normal/CN002_S_1280a102.png')
imgP=cv2.imread('C:/Users/personal/Documents/tutorial/proyecto/Test/Parkinson/CI127_S_0397a107.png')

plt.imshow(imgN)
plt.title('normal')
plt.show()

plt.imshow(imgP)
plt.title('parkinson')
plt.show()

print (imgN.shape)
print (imgP.shape)

print (imgN.max())
print (imgP.min())


print (imgN)# representacion cvectorial
print (np.prod(imgN.shape))
print (np.prod(imgP.shape))


imgN = cv2.resize(imgN,(100,100))
plt.imshow(imgN)
plt.title('Normal pixelada')
plt.show()

width = 100 #tamaño para reescalar imágen
num_classes = 2 #clases variable respuesta
trainpath = 'C:/Users/personal/Documents/tutorial/proyecto/Train/'
testpath = 'C:/Users/personal/Documents/tutorial/proyecto/Test/'

x_test, y_test = fn.img2data(testpath) #Run in test
x_train, y_train = fn.img2data(trainpath) #Run in train

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

joblib.dump(x_train, "x_train.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(x_test, "x_test.pkl")
joblib.dump(y_test, "y_test.pkl")