import package.classifiers.classification_techniques as classifiers
import package.segmentation.segmentation_techniques as segmenters
import package.plate_detection.localization_techniques as localizers


#__all__ = ['Segmenter', 'Classifier', 'localizer']

segmenter = segmenters.Segmenter
classifier = classifiers.Classifier
localize = localizers.Localize

'''
.............
def mix(image, loc, seg, clas):
    image = (localizer.localize(image, localization_strategy=loc))
    image = (segmenter.segment(image, segmentation_strategy=seg))
    name = classifier.classify(image, classification_strategy=clas)
    return name
'''
