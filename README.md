# AWS-Machine-Learning-Engineer-Capstone
AWS Machine Learning Enginner Nanodegree

TODO: replace all texts

# Definition

## Project Overview

Customers are the most important asset of an organization. Without customers, a business cannot 
survive or even exist. That is why an organization should have in its priorities the provision 
of the best customer experience and as result understand better its current and future customers. One way 
to better understand customers is by using techniques such as customer segmentation. Segmentation is an 
approach based on customers' traits, preferences, and behaviors that help businesses to better 
understand their customers better by dividing the customer base into distinct and internally homogeneous
groups [1]. Therefore, segmentation helps an organization to provide better products, offers, or solutions
and through that increase loyalty, brand reputation, competitive advantage, and growth.

## Problem Statement

This project goal is to segment customers based on their traits, preferences and behaviors, with that in mind this project tasks are diveded as
follers:
1. Download and upload Starbucks data to AWS S3
2. Preprocess data in AWS SageMaker
3. Train a unsupervised models to identify customer segments
4. Train a supervised classification model from unsupervised result to classify new customers
6. Deploy the supervised classification in a AWS SageMaker endpoint
7.  
The final model is expected to be able to classify new customers and provide the best promotional offers.


## Metrics

Accuracy is a common metric for binary classifiers; it takes into account both true positives and true
negatives with equal weight.

accuracy = dataset size / true positives + true negatives

# Analysis

The COCO (Common Objects in Context) dataset has hundreds of thousands of richly annotated
images; the annotations are not described here because only the images are used. More accurately, only
a subset of the images are used, the 2014 training images, as the COCO-Text 1.0 dataset has
annotations only for that particular subset. Of the 82,783 images, 63,686 contain text; altogether there
are 173,589 text instances, which is more than enough to train the classifier. The images are colored
and have around 600 * 400 pixels each.


## Exploratory Visualization

The plot below shows how the legible, English text annotations are distributed among the images. This
is helpful for predicting how balanced the classes will be after the images are segmented (see the Data
Preprocessing section).

Fig. 2 A plot showing how the legible, English text annotations are distributed among the images.

SHOW FIGURE HERE

Fig. 3 The following plot shows how the areas of the bounding boxes of the same annotations are
distributed.
This information can be helpful when trying to decide how big the image segments should be; if the
area of an annotation is larger than the area of a segment, then the annotation will be surely split apart,
which should be avoided for most of the annotations.

SHOW FIGURE HERE

## Algorithms and Techniques

The classifier is a Convolutional Neural Network, which is the state-of-the-art algorithm for most
image processing tasks, including classification. It needs a large amount of training data compared to
other approaches; fortunately, the COCO and COCO-Text datasets are big enough. The algorithm
outputs an assigned probability for each class; this can be used to reduce the number of false positives
using a threshold. (The tradeoff is that this increases the number of false negatives.)


## Benchmark 

To create an initial benchmark for the classifier, I used DIGITS (a web interface to the Caffe deep
learning library, to try multiple architectures. The “standard” LeNet architecture (Fig. 5) achieved the
best accuracy, around 0.8.
I couldn’t find a similar project that didn’t require special hardware, so the processing delay and
classification delay benchmarks had to be created without actual data:
❖ For the classification delay, my goal was going below 3 seconds, optimally below 200 ms.
❖ For the overall processing delay, my goal was reaching a delay below 6 seconds, optimally
below 500 ms.


# Methodology

## Data Preprocessing

The preprocessing done in the “Prepare data” notebook consists of the following steps:
1. The list of images is randomized
2. The images are divided into a training set and a validation set
3. The images are split into square shaped segments; random noise is used for padding
4. Each of the segments gets a label, which is “text” if the overlap between the segment and one
of the annotations is greater than a threshold, and “no-text” otherwise


Fig. 4 Segments from the “text” class produced with the final parameter settings. The segment size is
128 px, the overlap threshold is 500 px².

SHOW FIGURE HERE

## Implementation

The implementation process can be split into two main stages:
1. The classifier training stage
2. The application development stage

During the first stage, the classifier was trained on the preprocessed training data. This was done in a
Jupyter notebook (titled “Create and freeze graph”), and can be further divided into the following
steps:
1. Load both the training and validation images into memory, preprocessing them as described in
the previous section
2. Implement helper functions:
a. get_batch(...): Draws a random sample from the training/validation data
b. fill_feed_dict(...): Creates a feed_dict, which is a Python dictionary that contains all of
the data required for a single training step (a batch of images, their labels, and the
learning rate)
3. Define the network architecture and training parameters
4. Define the loss function, accuracy


## Results

Model Evaluation and Validation

During development, a validation set was used to evaluate the model.
The final architecture and hyperparameters were chosen because they performed the best among the
tried combinations.
For a complete description of the final model and the training process, refer to Figure 5 along with the
following list:
❖ The shape of the filters of the convolutional layers is 5*5.
❖ The first convolutional layer learns 32 filters, the second learns 64 filters.
❖ The convolutional layers have a stride of 2, so the resolution of the output matrices is half the
resolution of the input matrices.


# Conclusion

Reflection
The process used for this project can be summarized using the following steps:
1. An initial problem and relevant, public datasets were found
2. The data was downloaded and preprocessed (segmented)
3. A benchmark was created for the classifier
4. The classifier was trained using the data (multiple times, until a good set of parameters were
found)
5. The TensorFlow Android demo was adapted to run the classifier
6. The application was extended so that it can extract text from images using the Google Cloud
Vision API
7. Feeding the extracted text to the TTS system was implemented
I found steps 4 and 5 the most difficult, as I had to familiarize myself with the files of the TensorFlow
Android demo, which uses Bazel and the Android NDK, both of which were technologies that I was
not familiar with before the project.
As for the most interesting aspects of the project, I’m very glad that I found the COCO and
COCO-Text datasets, as I’m sure they’ll be useful for later projects/experiments. I’m also happy about
getting to use TensorFlow, as I believe it will be thedeep learning library in the future.
Improvement
To achieve the optimal user experience, using more capable hardware and moving the text extraction
9
process from the cloud to the device would be essential. This would reduce the processing time and
give access to the outputs of all of the modules of the text extraction pipeline, which would, in turn,
enable the following features:
❖ User-guided reading (e.g. read big text first, or read the text the user is pointing at)
❖ Better support for languages other than English
❖ Output filtering (e.g. ignore text smaller than some adjustable threshold)
❖ Passive text detection (auditory cue on text detection, perhaps with additional information
encoded in the tone and volume)
The user experience could also be improved significantly by using MXNet, which is a deep learning
library that is better optimized for mobile devices than TensorFlow. The speedup wouldn’t be enough
for running text extraction on the device, but it would reduce the classification delay significantly.
