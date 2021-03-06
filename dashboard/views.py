# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponse
from django.shortcuts import render
# from django.http import HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
from django import forms
from django.conf import settings
import os
import cv2
import package
import json
import time
from dashboard.toframe import video_framing

localization_strategies = {"Faster-Inception": 1}
segmentation_strategies = {"Faster-Inception": 1}
classification_strategies = {"MobileNet": 1}
# classification_strategies = {"MobileNet": 1, "VGG19": 1}
# classification_strategies = {"CNN": 1, "VGG16":1, "SVM": 1, "TemplateMatching": 1}
classification_unloaded = ['MobileNet:both', "SVM", "NASNetMobile", "DenseNet121", "DenseNet169", "DenseNet201",
                           "ResNet50", "NASNetLarge", "Xception", "InceptionV3", "InceptionResNetV2", "VGG16", "CNN",
                           "TemplateMatching"]
segmentation_unloaded = ['Faster-ResNet101', "Faster-Inception-ResNet", "RFCN-ResNet101"]
localization_unloaded = ['Faster-ResNet101', "Faster-Inception-ResNet", "PlateDetection"]

segmenter = package.segmenter(segmentation_strategies)
classifier = package.classifier(classification_strategies)
localize = package.localize(localization_strategies)

fs = FileSystemStorage()


def check_ext(file):
    file_type = file.content_type.split('/')[0]
    if len(file.name.split('.')) == 1:
        raise forms.ValidationError('File type is not supported')
    if file_type not in settings.TASK_UPLOAD_FILE_TYPES:
        raise forms.ValidationError('File type is not supported')

    return file_type


def handle_image(image_name, classification_type, segmentation_type, localization_type):
    path = os.path.join(settings.MEDIA_ROOT, image_name)
    note = ""
    class_detected = 1
    try:

        if (localization_type in [*localization_strategies, *localization_unloaded] and not (
                localization_type == "None")):

            load_object = False
            if localization_type in localization_unloaded:
                load_object = True

            try:
                localization_class = getattr(package.localizers, localization_type)
            except:
                localization_class = localization_type

            [[box, vehicle_image], class_detected, prob], _ = localize.localize(path,
                                                                                load_model=load_object,
                                                                                localization_strategy=localization_class)
            if vehicle_image is None:
                raise AssertionError
            else:
                vehicle_image = cv2.cvtColor(vehicle_image, cv2.COLOR_RGB2BGR)

            if box is None:
                raise AssertionError
            else:
                image = vehicle_image[box[2]:box[3], box[0]:box[1]]

            if segmentation_type == "None":
                note = str(time.strftime("%d/%m/%Y")) + " localization for " + image_name
                localized_name = "localization ___ " + fs.generate_filename(image_name)

                if not (localized_name == image_name):
                    fs.delete(image_name)

                directory = os.path.join(settings.MEDIA_ROOT, localized_name)
                localize.visualize(vehicle_image, directory, box, class_detected)

                return ["Localization by   " + localization_type, fs.url(localized_name), note]

        elif not (localization_type == "None"):
            raise ValueError("type not defined")

        else:
            image = cv2.imread(path)

    except Exception as e:
        print('%s (%s)' % (e, type(e)))
        return ["error", fs.url(image_name), "the localization module failed to extract the plate from image"]

    # --------------------segmentation handling-------------------------
    print("segmenting")
    try:
        if segmentation_type in [*segmentation_strategies, *segmentation_unloaded]:

            load_object = False
            if segmentation_type in segmentation_unloaded:
                load_object = True

            try:
                if int(class_detected) == 1:
                    segmentation_class = getattr(package.segmenters, segmentation_type)
                else:
                    load_object = True
                    segmentation_class = getattr(package.segmenters, "NewFCNResNet")
            except:
                if int(class_detected) == 1:
                    segmentation_class = segmentation_type
                else:
                    load_object = True
                    segmentation_class = "NewFCNResNet"

            [images, boxes, classes, scores, type_plate], _ = segmenter.segment(image,
                                                                                load_model=load_object,
                                                                                segmentation_strategy=segmentation_class)
            if classification_type == "None":
                note = type_plate
                segmented_name = "segmentation ___ " + fs.generate_filename(image_name)

                if not (segmented_name == image_name):
                    fs.delete(image_name)

                directory = os.path.join(settings.MEDIA_ROOT, segmented_name)
                segmenter.visualize(image, directory, boxes, classes)

                return ["Localization by " + localization_type + "   Segmentation by  " + segmentation_type,
                        fs.url(segmented_name),
                        note]

        else:
            raise ValueError

    except Exception as e:
        print('%s (%s)' % (e, type(e)))
        return ["error", fs.url(image_name), "the segmentation module failed to extract the plate from image"]

    # ---------------------classification handling---------------------------

    print("classifying")
    classification_class = None
    classification_object = None
    letters_note = ""
    note = " "

    try:
        if classification_type in [*classification_strategies, *classification_unloaded]:

            temp_classification_type = classification_type

            classifier.both = False

            load_object = False
            if classification_type in classification_unloaded:
                load_object = True

            type_splits = classification_type.split(':')
            if len(type_splits) > 1:
                classifier.both = True
                classification_type = type_splits[0]

            classification_class = getattr(package.classifiers, classification_type)

            for image_index in range(len(images)):
                [predicted_label, prob], classification_object = classifier.classify(images[image_index],
                                                                                     int(classes[image_index]),
                                                                                     load_model=load_object,
                                                                                     classification_strategy=classification_class,
                                                                                     get_object=True,
                                                                                     classification_object=classification_object)
                print(predicted_label)
                load_object = False
                # cv2.imshow("image", images[image_index])
                # cv2.waitKey(0)
                # cv2.imwrite("./" + image_index + ".jpg", images[image_index])
                # import scipy.misc
                # scipy.misc.imsave(str(image_index) +'.jpg', images[image_index])
                if classes[image_index] == 1:
                    letters_note += str(predicted_label) + " "
                else:
                    note += str(predicted_label)

                print(int(classes[image_index]))
                # print(note)

            cv2.destroyAllWindows()
            letters_note = letters_note[::-1]
            note += type_plate + " " + letters_note
            print(note)
            classification_type = temp_classification_type
            if classification_type not in classification_unloaded:
                classifier.append_classification_strategy(classification_class, classification_object)
            else:
                if not (classification_object is None):
                    del classification_object
        else:
            raise ValueError

    except Exception as e:

        if classification_object is not None and classification_class and classification_type not in classification_unloaded:
            classifier.append_classification_strategy(classification_class, classification_object)
        else:
            if not (classification_object is None):
                del classification_object

        print('%s (%s)' % (e, type(e)))

        return ["error", fs.url(image_name), "Memory error"]

    image_new_name = "classification ___ " + note + "- " + fs.generate_filename(image_name)
    new_path = os.path.join(settings.MEDIA_ROOT, image_new_name)
    os.rename(path, new_path)

    return ["Localization by " + localization_type +
             "   Segmentation by " + segmentation_type +
             "   Classification by " + temp_classification_type,
            fs.url(image_new_name),
            note]


def handle_video(filename, classification_type, segmentation_type, localization_type):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    images = video_framing(path)
    i = 0
    video_result = []
    for image in images:
        filename = fs.generate_filename(str(i) + ".jpg")
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, filename), image)
        value_array = handle_image(filename, classification_type, segmentation_type, localization_type)
        video_result.append(value_array)
    # call handle_image on each image

    return video_result


def handle_file(file, classification_type, segmentation_type, localization_type):
    type_file = check_ext(file)
    filename = fs.save(file.name, file)

    if type_file == "image":
        result = handle_image(filename, classification_type, segmentation_type, localization_type)
        result = [result]
    else:
        result = handle_video(filename, classification_type, segmentation_type, localization_type)

    return result


def index(request):
    try:
        if request.method == 'POST':
            images_url = []
            final_stage = "classification"
            note = ""
            data = []
            files = request.FILES.getlist('file')

            classification_type = request.POST['classification']
            segmentation_type = request.POST['segmentation']
            localization_type = request.POST['localization']

            for file in files:
                array_frames = handle_file(file,
                                           classification_type,
                                           segmentation_type,
                                           localization_type)
                for [final_stage, images_url, note] in array_frames:
                    data.append({"final_stage": final_stage, "images_url": images_url, "note": note})

            return HttpResponse(json.dumps(data))

    except Exception as e:
        print('%s (%s)' % (e, type(e)))
        return HttpResponse("error")

    view_dict = {"segmentation_strategies": [*segmentation_strategies, *segmentation_unloaded],
                 "classification_strategies": [*classification_strategies, *classification_unloaded],
                 "localization_strategies": [*localization_strategies, *localization_unloaded]}

    return render(request, 'dashboard/index.html', view_dict)


def show(request):
    array = []
    files = fs.listdir("./")
    for file in files[1]:
        try:
            splits = file.split("___")
            array.append([fs.url(file), splits[1], splits[0]])

        except Exception as e:
            print('%s (%s)' % (e, type(e)))

    return render(request, 'dashboard/show.html', {'image_list': array})
