import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from package.classifiers.classification_abstract import ClassificationAbstract
import os
import cv2

class CNN(ClassificationAbstract):
    def __init__(self):

        self.projectpath = os.path.join(os.getcwd(), "package", "classifiers")
        self.model_dir = os.path.join(os.getcwd(), "package", "classifiers", "models/CNN/")

        self.HEIGHT = 48
        self.WIDTH = 48

        self.epochs = 100
        self.batch_size = 32
        self.initial_epoch = 0

        self.nTrain = 0
        self.nVal = 0

        self.letters_model_path = "stable/letters-weights-improvement-*.hdf5"
        self.numbers_model_path = "stable/numbers-weights-improvement-*.hdf5"

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                self.letters_model = self.model_loader(self.letters_model_path)[0]
                self.numbers_model = self.model_loader(self.numbers_model_path)[0]


        self.letters_classes = {0: 'ا', 1: 'ب', 2: 'ج', 3: 'د', 4: 'ر', 5: 'س', 6: 'ص', 7: 'ط', 8: 'ع', 9: 'ف', 10: 'ق',
                                11: 'ل', 12: 'م', 13: 'ن', 14: 'ه', 15: 'و', 16: 'ى'}
        self.numbers_classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

    def get_classes(self):

        letters_dir = self.projectpath + '/plates_letters' + '/testset/'
        numbers_dir = self.projectpath + '/plates_numbers' + '/testset/'

        letters_classes = ImageDataGenerator().flow_from_directory(letters_dir).class_indices
        numbers_classes = ImageDataGenerator().flow_from_directory(numbers_dir).class_indices

        letters_classes = dict((v, k) for k, v in letters_classes.items())
        numbers_classes = dict((v, k) for k, v in numbers_classes.items())
        return letters_classes, numbers_classes

    def train(self, IMAGE_TYPE, new_train):

        if IMAGE_TYPE:
            num_classes = 17
            type = "letters"

        else:
            num_classes = 10
            type = "numbers"

        type_dir = '/plates_' + type

        train_dir = self.projectpath + type_dir + '/dataset/'
        validation_dir = self.projectpath + type_dir + '/testset/'

        self.nTrain = len(glob.glob(train_dir + '/**/*.*'))
        self.nVal = len(glob.glob(validation_dir + '/**/*.*'))

        train_generator, validation_generator = self.data_loader(train_dir, validation_dir)

        model = self.build_model(num_classes, new_train, type)

        self.run_model(type, model, train_generator,
                       validation_generator)

    def data_loader(self, train_dir, validation_dir):

        datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(self.HEIGHT, self.WIDTH),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle='shuffle')
        validation_generator = datagen.flow_from_directory(
            validation_dir,
            target_size=(self.HEIGHT, self.WIDTH),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)

        return train_generator, validation_generator

    def model_loader(self, model_path):

        models_url = glob.glob(os.path.join(self.model_dir, model_path))
        latest_model_path = sorted(models_url)[-1]
        print("model of cnn from" + latest_model_path)
        model = keras.models.load_model(latest_model_path)
        initial_epoch = int(latest_model_path.split(".")[0].split("-")[-2])

        return model, initial_epoch

    def build_model(self, num_classes, new_train, type):

        if (new_train):
            self.initial_epoch = 0
            input_shape = (self.HEIGHT, self.WIDTH, 3)
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
            return model

        model, self.initial_epoch = self.model_loader("volatile/" + type + "-weights-improvement-*.hdf5")
        return model

    def run_model(self, type, model, train_generator,
                  validation_generator):
        from keras.callbacks import ModelCheckpoint

        filepath = self.model_dir + "training/" + type + "-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                       min_delta=0,
                                                       patience=15,
                                                       verbose=0,
                                                       mode='auto')

        callbacks_list = [early_stopping, checkpoint]

        history = model.fit_generator(train_generator,
                                      epochs=self.epochs,
                                      callbacks=callbacks_list,
                                      validation_data=validation_generator,
                                      initial_epoch=self.initial_epoch)

    def validate(self, validation_generator, model, type, visulaize):

        if type == "letters":
            num_classes = 17
        else:
            num_classes = 10

        fnames = validation_generator.filenames
        ground_truth = validation_generator.classes
        label2index = validation_generator.class_indices
        nVal = len(ground_truth)

        validation_features = np.zeros(shape=(nVal, 48, 48, 3))
        validation_labels = np.zeros(shape=(nVal, num_classes))

        # Getting the mapping from class index to class label
        idx2label = dict((v, k) for k, v in label2index.iteritems())

        i = 0
        for inputs_batch, labels_batch in validation_generator:
            validation_features[i * self.batch_size: (i + 1) * self.batch_size] = inputs_batch
            validation_labels[i * self.batch_size: (i + 1) * self.batch_size] = labels_batch
            i += 1
            if i * self.batch_size >= nVal:
                break

        predictions = model.predict_classes(validation_features)
        prob = model.predict(validation_features)

        errors = np.where(predictions != ground_truth)[0]
        print("No of errors = {}/{}".format(len(errors), nVal))
        print("Acc = " + str(1 - len(errors) / (nVal * 1.0)))
        if (visulaize):
            self.visualize(errors, prob, idx2label, fnames, type)
        return errors, prob, idx2label, fnames

    def visualize(self, errors, prob, idx2label, fnames, type):

        for i in range(len(errors)):
            pred_class = np.argmax(prob[errors[i]])
            pred_label = idx2label[pred_class]

            print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
                fnames[errors[i]].split('/')[0],
                pred_label,
                prob[errors[i]][pred_class]))
            type_dir = '/plates_' + type
            validation_dir = self.projectpath + type_dir + '/testset/'
            original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
            plt.imshow(original)
            plt.show()

    def predict(self, image, type):
        keras.backend.set_session(self.session)

        image = cv2.resize(image, (self.WIDTH, self.HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) / 255

        image = np.array([image])
        if type == 1:
            with self.graph.as_default():
                with self.session.as_default():
                    prediction = self.letters_model.predict_classes(image)[0]
            prob = self.letters_model.predict(image)
            pred_label = self.letters_classes[prediction]

        elif type == 2:
            with self.graph.as_default():
                with self.session.as_default():
                    prediction = self.numbers_model.predict_classes(image)[0]
            prob = self.numbers_model.predict(image)
            pred_label = self.numbers_classes[prediction]
        else:
            raise TypeError
        probability = prob[0][prediction]
        return pred_label, probability
