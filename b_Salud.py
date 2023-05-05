import numpy as np
import cv2 ### para leer imagenes jpg
from matplotlib import pyplot as plt ## para gráfciar imágnes
import a_Funciones as fn #### funciones personalizadas, carga de imágenes
import joblib ### para descargar array

# Imágenes
img1 = cv2.imread('Datos/train/AD/AD002_S_0816a076.png')
img2 = cv2.imread('Datos/test/CN/CN082_S_0640a089.png')

# Gráfica imágenes
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.title('Alzhaimer')

plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title('Normal');

# representación numérica de imágenes ####

img1.shape # tamaño de imágenes
img1.max() # máximo valor de intensidad en un pixel
img1.min() # mínimo valor de intensidad en un pixel

np.prod(img1.shape) ### 130 mil valores que representan la imagen (columnas)

# Se reescalan img1
img1 = cv2.resize(img1 ,(100,100))
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.title('Alzhaimer')

img2 = cv2.resize(img2 ,(100,100))
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title('Normal');

# Rutas para función
trainpath = 'Datos/train/'
testpath = 'Datos/test/'

# Se usa función para obtener la mayor dimension de las imágenes.
# Se deberían tener más observaciones que columnas.
fn.dimen_data(trainpath)
fn.dimen_data(testpath)

# Cargar todas las imágenes y reducir su tamaño
x_train, y_train = fn.img2data(trainpath) 
x_test, y_test = fn.img2data(testpath) 

#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

np.prod(x_train[1000].shape) # cada imagen está representada con un array de 30.000 datos
x_train.shape
x_test.shape
y_train.shape
y_test.shape

# Se guardan los archivos para ser usados en la modelación
joblib.dump(x_train, "x_train.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(x_test, "x_test.pkl")
joblib.dump(y_test, "y_test.pkl")