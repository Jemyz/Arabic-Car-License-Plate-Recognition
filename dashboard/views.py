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

localization_strategies = {"PlateDetection": 1}
segmentation_strategies = {"Inception": 1}
# classification_strategies = {"CNN": 1, "ImageNet": 1, "Svm": 1, "TemplateMatching": 1}
classification_strategies = {"TemplateMatching": 1}

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
    # --------------------localization handling----------------------

    if localization_type in localization_strategies and not (localization_type == "none"):
        localization_class = getattr(package.localizers, localization_type)
        [box, vehicle_image], localization_object = localize.localize(path, localization_strategy=localization_class,
                                                                      get_object=True)

        if vehicle_image is not None:
            vehicle_image = cv2.cvtColor(vehicle_image, cv2.COLOR_RGB2BGR)

        if box is None:
            return ["error", "", "no plate found"]
        else:
            image = vehicle_image[box[2]:box[3], box[0]:box[1]]

        if segmentation_type == "None":
            note = str(time.strftime("%d/%m/%Y")) + " localization for " + image_name
            localized_name = "localization - " + fs.generate_filename(image_name)

            if not (localized_name == image_name):
                fs.delete(image_name)

            directory = os.path.join(settings.MEDIA_ROOT, localized_name)
            localize.visualize(vehicle_image, directory, box,
                               localization_strategy=localization_class,
                               localization_object=localization_object)
            return ["localization", fs.url(localized_name), note]
        localize.append_localization_strategy(localization_class, localization_object)
    else:
        raise ValueError

    # --------------------segmentation handling-------------------------

    if segmentation_type in segmentation_strategies:
        segmentation_class = getattr(package.segmenters, segmentation_type)
        [images, boxes, classes, scores], segmentation_object = segmenter.segment(image,
                                                                                  segmentation_strategy=segmentation_class,
                                                                                  get_object=True)

        if classification_type == "None":
            note = str(time.strftime("%d/%m/%Y")) + " segmentation for " + image_name
            segmented_name = "segmentation - " + fs.generate_filename(image_name)

            if not (segmented_name == image_name):
                fs.delete(image_name)

            directory = os.path.join(settings.MEDIA_ROOT, segmented_name)
            segmenter.visualize(image, directory, boxes, classes, scores,
                                segmentation_object=segmentation_object,
                                segmentation_strategy=segmentation_class)

            return ["segmentation", fs.url(segmented_name), note]

        segmenter.append_segmentation_strategy(segmentation_class, segmentation_object)
    else:
        raise ValueError

    # ---------------------classification handling---------------------------
    print("classifying")

    if classification_type in classification_strategies:
        class_object = None
        classification_class = getattr(package.classifiers, classification_type)

        for image_index in range(len(images)):
            [predicted_label, probability], class_object = classifier.classify(images[image_index],
                                                                               int(classes[image_index]),
                                                                               classification_strategy=classification_class,
                                                                               get_object=True,
                                                                               classification_object=class_object)

            #cv2.imshow("image", images[image_index])
            #cv2.waitKey()
            note += str(predicted_label)
            # print(int(classes[image_index]))
            print(note)

        classifier.append_classification_strategy(classification_class, class_object)
    else:
        raise ValueError

    return ["classification", fs.url(image_name), note]


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

    view_dict = {"segmentation_strategies": segmentation_strategies.keys(),
                 "classification_strategies": classification_strategies.keys(),
                 "localization_strategies": localization_strategies.keys()}

    return render(request, 'dashboard/index.html', view_dict)


def show(request):
    fs = FileSystemStorage()
    array = []
    files = fs.listdir("./")
    for file in files[1]:
        try:
            splits = file.split("-")
            array.append([fs.url(file), splits[1], splits[0]])
        except:
            array.append([fs.url(file)])
    return render(request, 'dashboard/show.html', {'image_list': array})
