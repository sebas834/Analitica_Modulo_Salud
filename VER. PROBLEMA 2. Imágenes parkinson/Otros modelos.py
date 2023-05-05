
import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier 

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


#### como tenemos overfitting, se hará más sencillo en modelo #########
                  




###11111111111111111111111111111111111111111111111111111111111111111111111111111111111111
Pruebas1=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

Pruebas1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])# binaru es la funcion de perdida

#####Entrenar el modelo usando el optimizador y arquitectura definidas #
Pruebas1.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

#########Evaluar el modelo ####################
test_loss, test_acc, test_auc, test_recall, test_precision = Pruebas1.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)
######### EL MODLEO MEJORA CONSIDERABLEMENTE######






#####################22222222222222222222############################################
Pruebas2=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

Pruebas2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])# binaru es la funcion de perdida

#####Entrenar el modelo usando el optimizador y arquitectura definidas #
Pruebas2.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

#########Evaluar el modelo ####################
test_loss, test_acc, test_auc, test_recall, test_precision = Pruebas2.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)
######### EL MODLEO SE MANTIENE######






#############3333333333333333333333333333333333333333##################33333333333333333333333
Pruebas3=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

Pruebas3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])# binaru es la funcion de perdida

#####Entrenar el modelo usando el optimizador y arquitectura definidas #
Pruebas3.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

#########Evaluar el modelo ####################
test_loss, test_acc, test_auc, test_recall, test_precision = Pruebas3.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)
######### EL MODLEO MEJORA######






##########444444444444444444444444444444444444444444444444444####################################3
######### HIPERPARAMETROS AL MODLEO QUE MEJORA CONSIDERABLEMENTE######
import keras_tuner as kt

def hiperparámetros1(hp): #### al modelo original. siempre se debe crear una funcion para cada modelo
    
    dropout_rate=hp.Float('DO', min_value=0.1, max_value= 0.4, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.01, max_value=0.05, step=0.001)
    ####hp.Int
    ####hp.Choice
    

    Pruebas1=tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=x_train.shape[1:]), 
      tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
      tf.keras.layers.Dropout(dropout_rate),
      tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
      tf.keras.layers.Dropout(dropout_rate),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
   
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
   
    Pruebas1.compile(
        optimizer=opt, loss="binary_crossentropy", metrics=["AUC"],
    )
    return Pruebas1


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

#################### Mejor redes ##############

joblib.dump(fc_best_model, 'fc_model.pkl')

########## SE MANTIENE DENTRO DE 67%  #####################33









##########55555555555555555555555555555555555555555555555555555555555555555555##############################
#####se probará con redes convolucionales#########
########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier  ### para analizar modelo

Convolucional = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
Convolucional.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

# Train the model for 10 epochs
Convolucional.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))
####### genera un sobreajuste mayor ################################ Overfitting
 