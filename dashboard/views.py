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

localization_strategies = {"ObjectDetection": 1}
segmentation_strategies = {"Inception": 1}
classification_strategies = {"CNN": 1, "SVM": 1, "TemplateMatching": 1}
# classification_strategies = {"CNN": 1, "VGG16":1, "SVM": 1, "TemplateMatching": 1}
classification_unloaded = ["VGG16"]
segmentation_unloaded = []
localization_unloaded = []

segmenter = package.segmenter(segmentation_strategies)
classifier = package.classifier(classification_strategies)
localize = package.localize(localization_strategies)


def check_ext(file):
    file_type = file.content_type.split('/')[0]
    if len(file.name.split('.')) == 1:
        raise forms.ValidationError('File type is not supported')
    if file_type not in settings.TASK_UPLOAD_FILE_TYPES:
        raise forms.ValidationError('File type is not supported')

    return file_type


def handle_image(image_name, classification_type, segmentation_type, localization_type, fs):
    path = os.path.join(settings.MEDIA_ROOT, image_name)
    note = ""
    localization_object = None
    localization_class = None
    segmentation_object = None
    segmentation_class = None
    classification_object = None
    classification_class = None

    # --------------------localization handling----------------------
    try:

        if (localization_type in localization_strategies or localization_type in localization_unloaded) and not (
                localization_type == "None"):
            load_object = False
            if localization_type in localization_unloaded:
                load_object = True

            localization_class = getattr(package.localizers, localization_type)
            [[box, vehicle_image], class_detected, prob], localization_object = localize.localize(path,
                                                                                                  load_model=load_object,
                                                                                                  localization_strategy=localization_class,
                                                                                                  get_object=True)

            if vehicle_image is not None:
                vehicle_image = cv2.cvtColor(vehicle_image, cv2.COLOR_RGB2BGR)

            if box is None:
                raise AssertionError
            else:
                image = vehicle_image[box[2]:box[3], box[0]:box[1]]

            if segmentation_type == "None":
                note = str(time.strftime("%d/%m/%Y")) + " localization for " + image_name
                localized_name = "localization - " + fs.generate_filename(image_name)

                if not (localized_name == image_name):
                    fs.delete(image_name)

                directory = os.path.join(settings.MEDIA_ROOT, localized_name)
                localize.visualize(vehicle_image, directory, box, class_detected)

                if localization_type not in localization_unloaded:
                    localize.append_localization_strategy(localization_class, localization_object)

                return ["localization", fs.url(localized_name), note]

            if localization_type not in localization_unloaded:
                localize.append_localization_strategy(localization_class, localization_object)

        elif not (localization_type == "None"):
            raise ValueError
        else:
            image = cv2.imread(path)

    except Exception as e:

        if localization_object is not None and localization_class:
            localize.append_localization_strategy(localization_class, localization_object)
        fs.delete(image_name)
        print('%s (%s)' % (e, type(e)))
        return ["error", "", "error happened while detecting the plate please try again"]

    # --------------------segmentation handling-------------------------
    try:
        if segmentation_type in segmentation_strategies or segmentation_type in segmentation_unloaded:
            load_object = False
            if segmentation_type in segmentation_unloaded:
                load_object = True

            segmentation_class = getattr(package.segmenters, segmentation_type)
            [images, boxes, classes, scores], segmentation_object = segmenter.segment(image,
                                                                                      load_model=load_object,
                                                                                      segmentation_strategy=segmentation_class,
                                                                                      get_object=True)

            if classification_type == "None":
                note = str(time.strftime("%d/%m/%Y")) + " segmentation for " + image_name
                segmented_name = "segmentation - " + fs.generate_filename(image_name)

                if not (segmented_name == image_name):
                    fs.delete(image_name)

                directory = os.path.join(settings.MEDIA_ROOT, segmented_name)
                segmenter.visualize(image, directory, boxes, classes)

                if segmentation_type not in segmentation_unloaded:
                    segmenter.append_segmentation_strategy(segmentation_class, segmentation_object)

                return ["segmentation", fs.url(segmented_name), note]
            if segmentation_type not in segmentation_unloaded:
                segmenter.append_segmentation_strategy(segmentation_class, segmentation_object)
        else:
            raise ValueError

    except Exception as e:

        if segmentation_object is not None and segmentation_class:
            segmenter.append_segmentation_strategy(segmentation_class, segmentation_object)
        fs.delete(image_name)
        print('%s (%s)' % (e, type(e)))
        return ["error", "", "error happened while segmenting the plate please try again"]

    # ---------------------classification handling---------------------------
    print("classifying")
    try:
        if classification_type in classification_strategies or classification_type in classification_unloaded:
            classification_object = None
            classification_class = getattr(package.classifiers, classification_type)
            load_object = False

            if classification_type in classification_unloaded:
                load_object = True

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
                # cv2.waitKey()
                # cv2.imwrite("./" + image_index + ".jpg", images[image_index])
                # import scipy.misc
                # scipy.misc.imsave(str(image_index) +'.jpg', images[image_index])
                note += str(predicted_label).strip()
                # print(int(classes[image_index]))
                # print(note)
            print(note)

            if classification_type not in classification_unloaded:
                classifier.append_classification_strategy(classification_class, classification_object)
        else:
            raise ValueError

    except Exception as e:

        if classification_object is not None and classification_class:
            classifier.append_classification_strategy(classification_class, classification_object)
        fs.delete(image_name)
        print('%s (%s)' % (e, type(e)))
        return ["error", "", "error happened while classifying the plate please try again"]

    image_new_name = "classification - " + note + "- " + fs.generate_filename(image_name)
    new_path = os.path.join(settings.MEDIA_ROOT, image_new_name)
    os.rename(path, new_path)

    return ["classification", fs.url(image_new_name), note]


def handle_video(filename, classification_type, segmentation_type, localization_type, fs):
    # frame video and get images
    # call handle_image on each image

    return filename


def handle_file(file, classification_type, segmentation_type, localization_type, fs):
    type_file = check_ext(file)
    filename = fs.save(file.name, file)

    if type_file == "image":
        result = handle_image(filename, classification_type, segmentation_type, localization_type, fs)
    else:
        result = handle_video(filename, classification_type, segmentation_type, localization_type, fs)

    return result


def index(request):
    fs = FileSystemStorage()

    try:
        if request.method == 'POST':
            images_url = []
            final_stage = "classification"
            note = ""

            files = request.FILES.getlist('file')

            classification_type = request.POST['classification']
            segmentation_type = request.POST['segmentation']
            localization_type = request.POST['localization']

            for file in files:
                final_stage, images_url, note = handle_file(file, classification_type, segmentation_type,
                                                            localization_type, fs)

            data = {"final_stage": final_stage, "images_url": images_url, "note": note}
            return HttpResponse(json.dumps(data))

    except Exception as e:
        print('%s (%s)' % (e, type(e)))
        return HttpResponse("error")

    view_dict = {"segmentation_strategies": [*segmentation_strategies, *segmentation_unloaded],
                 "classification_strategies": [*classification_strategies, *classification_unloaded],
                 "localization_strategies": [*localization_strategies, *localization_unloaded]}

    return render(request, 'dashboard/index.html', view_dict)


def show(request):
    fs = FileSystemStorage()
    array = []
    files = fs.listdir("./")
    for file in files[1]:
        try:
            splits = file.split("-")
            array.append([fs.url(file), splits[1], splits[0]])

        except Exception as e:
            array.append([fs.url(file)])
            print('%s (%s)' % (e, type(e)))

    return render(request, 'dashboard/show.html', {'image_list': array})
