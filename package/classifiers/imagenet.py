from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
import glob
import pickle
from package.classifiers.classification_abstract import ClassificationAbstract
import cv2


from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import Xception
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications import NASNetLarge
from keras.applications import ResNet50
from keras.applications import DenseNet201
from keras.applications import DenseNet169
from keras.applications import DenseNet121
from keras.applications import MobileNet  # 128
from keras.applications import NASNetMobile  # 224


class ImageNet(ClassificationAbstract):

    def __init__(self,feature_class):

        self.projectpath = os.path.join(os.getcwd(), "package", "classifiers")
        self.model_dir = os.path.join(os.getcwd(), "package", "classifiers", "models/ImageNet/")

        self.size_dict = {MobileNet: [128, 128], NASNetMobile: [224, 224],
                          DenseNet121: [221, 221], DenseNet169: [221, 221],
                          DenseNet201: [221, 221], ResNet50: [197, 197],
                          NASNetLarge: [331, 331], Xception: [71, 71],
                          InceptionV3: [139, 139], InceptionResNetV2: [139, 139],
                          VGG16: [48, 48], VGG19: [48, 48]}

        self.HEIGHT = self.size_dict[feature_class][0]
        self.WIDTH = self.size_dict[feature_class][1]

        self.epochs = 100
        self.batch_size = 64
        self.initial_epoch = 0

        self.nTrain = 0
        self.nVal = 0

        self.letters_model_path = "stable/letters-weights-improvement-*.hdf5"
        self.numbers_model_path = "stable/numbers-weights-improvement-*.hdf5"

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                self.feature_model = feature_class(include_top=False,
                                                   weights='imagenet',
                                                   input_shape=(self.HEIGHT, self.WIDTH, 3))

                self.feature_model_output_shape = self.feature_model.get_output_shape_at(-1)
                self.letters_model = self.model_loader(self.letters_model_path)[0]
                self.numbers_model = self.model_loader(self.numbers_model_path)[0]

        # self.vgg_conv._make_predict_function()

        # self.vgg_conv.predict(np.zeros((1, 48, 48, 3)))
        # self.letters_model.predict(np.zeros((1, 25088)))
        # self.numbers_model.predict(np.zeros((1, 25088)))

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

        train_features_dir = "pickle/" + type + "/train/"
        validation_features_dir = "pickle/" + type + "/validation/"

        train_features, train_labels = self.generate_vgg_features(
            self.nTrain, num_classes, train_generator, train_features_dir)
        validation_features, validation_labels = self.generate_vgg_features(
            self.nVal, num_classes, validation_generator, validation_features_dir)

        model = self.build_model(num_classes, new_train, type)
        self.run_model(type, model, train_features,
                       train_labels, validation_features, validation_labels)

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

    def generate_vgg_features(self, num, num_classes,
                              generator, feature_dir):

        features = np.zeros(shape=(num, 7, 7, 512))
        labels = np.zeros(shape=(num, num_classes))

        if (os.path.isfile(feature_dir + "features.pickle")
                and os.path.isfile(feature_dir + "labels.pickle")):

            with open(feature_dir + "features.pickle", 'rb') as handle:
                features = pickle.load(handle, encoding='latin1')
            with open(feature_dir + "labels.pickle", 'rb') as handle:
                labels = pickle.load(handle, encoding='latin1')

        else:

            i = 0
            for inputs_batch, labels_batch in generator:

                features_batch = self.feature_model.predict(inputs_batch)
                features[i * self.batch_size: (i + 1) * self.batch_size] = features_batch
                labels[i * self.batch_size: (i + 1) * self.batch_size] = labels_batch
                i += 1
                if i * self.batch_size >= num:
                    break
            features = np.reshape(features, (num, 7 * 7 * 512))

            with open(feature_dir + "features.pickle", 'wb') as handle:
                pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(feature_dir + "labels.pickle", 'wb') as handle:
                pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return features, labels

    def model_loader(self, model_path):

        models_url = glob.glob(os.path.join(self.model_dir, model_path))
        latest_model_path = sorted(models_url)[-1]
        print("image net model from " + latest_model_path)
        model = keras.models.load_model(latest_model_path)
        initial_epoch = int(latest_model_path.split(".")[0].split("-")[-2])

        return model, initial_epoch

    def build_model(self, num_classes, new_train, type):

        if (new_train):
            self.initial_epoch = 0
            model = models.Sequential()
            model.add(layers.Dense(128, activation='relu', input_dim=7 * 7 * 512))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(num_classes, activation='softmax'))

            model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                          loss='categorical_crossentropy',
                          metrics=['acc'])
            return model

        model, self.initial_epoch = self.model_loader("volatile/" + type + "-weights-improvement-*.hdf5")
        return model

    def run_model(self, type, model, train_features,
                  train_labels, validation_features, validation_labels):
        from keras.callbacks import ModelCheckpoint

        filepath = self.model_dir + "training/" + type + "-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                       min_delta=0,
                                                       patience=30,
                                                       verbose=0,
                                                       mode='auto')

        callbacks_list = [early_stopping, checkpoint]

        history = model.fit(train_features,
                            train_labels,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            callbacks=callbacks_list,
                            initial_epoch=self.initial_epoch,
                            validation_data=(validation_features, validation_labels))

    def validate(self, validation_generator, model, validation_features, type, visulaize):

        fnames = validation_generator.filenames

        ground_truth = validation_generator.classes

        label2index = validation_generator.class_indices
        nVal = len(ground_truth)

        # Getting the mapping from class index to class label
        idx2label = dict((v, k) for k, v in label2index.items())

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

    def predict(self, image, image_type):
        keras.backend.set_session(self.session)

        image = cv2.resize(image, (self.WIDTH, self.HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) / 255

        features = np.zeros(shape=(1, self.feature_model_output_shape[1],
                                   self.feature_model_output_shape[2], self.feature_model_output_shape[3]))

        with self.graph.as_default():
            with self.session.as_default():
                features[0:1] = self.feature_model.predict(np.array([image]))

        features = np.reshape(features, (1, self.feature_model_output_shape[1] *
                                         self.feature_model_output_shape[2] * self.feature_model_output_shape[3]))

        if image_type == 1:
            with self.graph.as_default():
                with self.session.as_default():
                    prediction = self.letters_model.predict_classes(features)[0]
            prob = self.letters_model.predict(features)
            pred_label = self.letters_classes[prediction]

        elif image_type == 2:
            with self.graph.as_default():
                with self.session.as_default():
                    prediction = self.numbers_model.predict_classes(features)[0]
            prob = self.numbers_model.predict(features)
            pred_label = self.numbers_classes[prediction]
        else:
            raise TypeError
        probability = prob[0][prediction]
        return pred_label, probability

