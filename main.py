from res_net import ResNet50
import keras.backend as K
import data_service
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot



K.set_image_data_format('channels_last')
K.set_learning_phase(1)

model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = data_service.load_dataset()

X_train, Y_train, X_test, Y_test = data_service.preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

model.fit(X_train, Y_train, epochs=2, batch_size=32)

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

model.summary()

plot_model(model, to_file='plots/model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))
