import utils
import cv2
from skimage import feature


def extract_hog_features():
    print("Extracting HOG features")
    X = []
    Y = []
    for i in range(1, utils.batch_num + 1):
        print("\tbatch " + str(i))
        data_norm, labels = utils.get_data_from_batch(i)

        for k in range(len(labels)):
            H = process_HOG(data_norm[k])
            X.append(H)
            Y.append(labels[k])

    XT = []
    YT = []
    print("\ttest batch")
    data_norm, labels = utils.get_data_from_test_batch()
    for k in range(len(labels)):
        H = process_HOG(data_norm[k])
        XT.append(H)
        YT.append(labels[k])

    return X, Y, XT, YT


def process_HOG(image):
    # before using HOG we need to change image from RGB to Gray
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    H = feature.hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2),
                    transform_sqrt=True)
    return H


utils.check_download_dataset()
utils.check_extract_dataset()

X, Y, XT, YT = extract_hog_features()
utils.run_svm(X, Y, XT, YT)
