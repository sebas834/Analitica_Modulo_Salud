import numpy as np
import joblib ### para cargar array
import tensorflow as tf
from sklearn import metrics 
import keras_tuner as kt

### se cargan los archivos

x_train = joblib.load('x_train.pkl')
y_train = joblib.load('y_train.pkl')
x_test = joblib.load('x_test.pkl')
y_test = joblib.load('y_test.pkl')

x_train.shape
x_test.shape

# normalizar variables 
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train /=255 ### escalaro para que quede entre 0 y 1
x_test /=255
y_train.shape
y_test.shape

# Definir arquitectura de la red neuronal e instanciar el modelo 

fc_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

fc_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy','AUC'])
fc_model.fit(x_train, y_train, batch_size=20, epochs=10, validation_data=(x_test, y_test))
test_loss, test_acc, test_auc = fc_model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

#################################################################################
#################################################################################
dropout_rate = 0.3 ## porcentaje de neuronas que elimina
reg_strength = 0.001

fc_model1 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

fc_model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC'])
fc_model1.fit(x_train, y_train, batch_size=20, epochs=10, validation_data=(x_test, y_test))
test_loss, test_acc, test_auc = fc_model1.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

#################################################################################
#################################################################################
#################################################################################
#################################################################################


# #########Evaluar el modelo ####################
test_loss, test_acc, test_auc = fc_model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

pred_test=(fc_model1.predict(x_test) > 0.5).astype('int')

cm=metrics.confusion_matrix(y_test,pred_test, labels=[0,1])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
disp.plot()

##########################################################
################ Red convolucional ###################
##########################################################

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

# Train the model for 10 epochs
cnn_model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

test_loss, test_acc, test_auc = cnn_model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

#################################################################################
#################################################################################

##### función con definicion de hiperparámetros a afinar

def build_model(hp):
    
    dropout_rate=hp.Float('DO', min_value=0.1, max_value= 0.4, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.0005, step=0.0001)
    ####hp.Int
    ####hp.Choice
    

    model11=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
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
   
    model11.compile(
        optimizer=opt, loss="binary_crossentropy", metrics=["AUC"],
    )
    return model11


###########
hp = kt.HyperParameters()
build_model(hp)

tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=False, ## solo evalúe los hiperparámetros configurados
    objective=kt.Objective("val_auc", direction="max"),
    max_trials=10,
    overwrite=True,
    directory="C:/Users/SEBASTIAN/Desktop",
    project_name="ensay", 
)

tuner.search(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=100)

fc_best_model = tuner.get_best_models(num_models=1)[0]
tuner.results_summary()

joblib.dump(fc_best_model, 'fc_model.pkl')
