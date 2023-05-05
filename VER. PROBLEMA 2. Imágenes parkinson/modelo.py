import numpy as np
import joblib ### para cargar array
import keras_tuner as kt
########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier  ### para analizar modelo

x_train = joblib.load('x_train.pkl')
y_train = joblib.load('y_train.pkl')
x_test = joblib.load('x_test.pkl')
y_test = joblib.load('y_test.pkl')


x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train /=255 ### escalaro para que quede entre 0 y 1
x_test /=255
print(np.product(x_train[1].shape))


y_train.shape
y_test.shape

########################NEURONA INICIAL########################

modelo1=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])# binaru es la funcion de perdida
modelo1.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

#########EVALUAR EL MODELO ####################
test_loss, test_acc, test_auc, test_recall, test_precision = modelo1.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)


###sobreajuste ##########################################
######################overfitting
############porbar otro modelo, más simple. Estrategia para mejorar el modelo.
######### VER ARCHIVO OTROS MODELOS#################




##########      MATRIZ DE CONFUSIÓN    1     #############
pred_test=(modelo1.predict(x_test) > 0.50).astype('int')

cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Parkinson', 'Normal'])

print(metrics.classification_report(y_test, pred_test))
disp.plot()





## regularizacion1
## regularizacion1
## regularizacion1
## regularizacion1
## regularizacion1

dropout_rate = 0.25 
modelo1R1=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

##### configura el optimizador y la función para optimizar ##############

modelo1R1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])# binaru es la funcion de perdida
modelo1R1.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

#####Entrenar el modelo usando el optimizador y arquitectura definidas #########
modelo1R1.fit(x_train, y_train, batch_size=100, epochs=7, validation_data=(x_test, y_test))

test_loss, test_acc, test_auc, test_recall, test_precision = modelo1R1.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)
####### modelo1R1  mejora en un 13 % ############# mejora en entrenamiento y ajuste 63%
##########################################################################
###########################################################################






## regularizacion2
## regularizacion2
## regularizacion2
## regularizacion2
## regularizacion2


reg_strength = 0.003
###########Estrategias a usar: regilarization usar una a la vez para ver impacto
dropout_rate = 0.25 ## porcentaje de neuronas que utiliza 
modelo1R2=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(4, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
modelo1R2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])# binaru es la funcion de perdida
modelo1R2.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))


test_loss, test_acc, test_auc, test_recall, test_precision = modelo1R2.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)
####### modelo1R2  mejora en un 5 % ############# mejora en entrenamiento y ajuste 68%
##########################################################################
###########################################################################

##########    MATRIZ DE CONFUSIÓN   2     #############
pred_test=(modelo1R2.predict(x_test) > 0.50).astype('int')

cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Parkinson', 'Normal'])

print(metrics.classification_report(y_test, pred_test))
disp.plot()





###############################                         #######################333
######### definir función para encontrar la mejor combinación de los parámetros   #######
###############################                         ################################3
def hiperparámetros1(hp): #### al modelo original. siempre se debe crear una funcion para cada modelo
    
    dropout_rate=hp.Float('DO', min_value=0.1, max_value= 0.4, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.01, max_value=0.05, step=0.0001)
    ####hp.Int
    ####hp.Choice
    

    modelo1R2=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(4, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
   
    modelo1R2.compile(
        optimizer=opt, loss="binary_crossentropy", metrics=["AUC"],
    )
    return modelo1R2


###########
hp = kt.HyperParameters()
hiperparámetros1(hp)

tuner = kt.RandomSearch(
    hypermodel=hiperparámetros1,
    hyperparameters=hp,
    tune_new_entries=False, ## solo evalúe los hiperparámetros configurados
    objective=kt.Objective("val_auc", direction="max"),
    max_trials=10,
    overwrite=True,
    directory="proyecto",
    project_name="HiperRedes", 
)

tuner.search(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=100)

fc_best_model = tuner.get_best_models(num_models=1)[0]
tuner.results_summary()

#################### Mejor red  ##############

joblib.dump(fc_best_model, 'fc_model.pkl')

##### el modleo no mejora significativamente ########
#### inclusive al se cambiar intervalos de los parámetros




 
#####se probará con redes convolucionales#########
########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier  ### para analizar modelo

Convolucional1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
Convolucional1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

# Train the model for 10 epochs
Convolucional1.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))
####### genera un sobreajuste mayor ################################ Overfitting



############porbar otro modelo, más simple. Estrategia para mejorar el modelo 


Convolucional2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
Convolucional2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

# Train the model for 10 epochs
Convolucional2.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))
####### genera un sobreajuste, pero mejora la marca de ajuste.