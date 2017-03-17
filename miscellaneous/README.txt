# Miscellaneous Implementations

This folder contains various algorithms which were tested for this problem. Please run all code from the root of this package.

To run random predictor, do:
    python miscellaneous/random_predictor.py

To run SVM, do:
    python miscellaneous/svm.py

If you wish to run Darknet19, you will need to install and compile it, as explained here: https://pjreddie.com/darknet/install/

You will also need to download the pretrained Darknet19 weights and run the label_images.py script which will save all the images in the correct folders. Then, run Darnet19 according to the instructions.

Some changes to the main execution script were made to save the output to a file instead of print it to a console, so that code cannot be run. It was excluded because it will not compile without the entire Darknet package. However, the results are in the results directory and can be processed with:

    python miscellaneous/darknet/post_processing.py
