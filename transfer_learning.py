import utils
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from keras.preprocessing.image import ImageDataGenerator


def check_extract_inception_features():
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(139, 139, 3))
    return check_extract_data_features(model)


def check_extract_inceptionresnet_features():
    model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(139, 139, 3))
    return check_extract_data_features(model)


def check_make_data_augmentation():
    if not os.path.exists('augmented_features_train.npz') or not os.path.exists('augmented_features_test.npz'):
        print("Data augmentation")
        model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(139, 139, 3))

        YP = []
        Y = []
        # image augmentation with only horizontal flip
        datagen = ImageDataGenerator(horizontal_flip=True)
        for i in range(1, utils.batch_num + 1):
            print("\tbatch " + str(i))
            data_norm, labels = utils.get_data_from_batch(i)

            prepared_ims = prepare_images(data_norm, model)
            datagen.fit(prepared_ims)

            # generate 250 augmented images in each batch
            for x_batch, y_batch in datagen.flow(prepared_ims, labels, batch_size=250,
                                                 seed=0):  # seed set to make experiments repeatable
                prepared_ims = np.concatenate((prepared_ims, x_batch), axis=0)
                labels = np.concatenate((np.asarray(labels), y_batch), axis=0)
                break
            Y.append(labels)
            predicted = model.predict(prepared_ims)
            YP.append(predicted)

            np.savez('augmented_features_train', train_features=YP)
            np.savez('augmented_labels_train', labels=Y)
    else:
        print("Loading augmented data")

    augmented_train_features = np.load('augmented_features_train.npz')['train_features']
    augmented_train_features = augmented_train_features.reshape((-1, model.output_shape[1]))
    augmented_train_labels = np.load('augmented_labels_train.npz')['labels']
    augmented_train_labels = augmented_train_labels.flatten()

    return augmented_train_features, augmented_train_labels


def prepare_images(data_norm, model):
    # scaling image size to match model's input
    scaled_ims = np.array(
        [scipy.misc.imresize(data_norm[k], model.input_shape[1:]) for k in range(0, len(data_norm))]).astype(
        'float32')
    # then make preprocessing required by model
    return preprocess_input(scaled_ims)


def extract_labels():
    Y = []
    for i in range(1, utils.batch_num + 1):
        _, labels = utils.get_data_from_batch(i)
        Y.extend(labels)

    YT = []
    _, labels = utils.get_data_from_test_batch()
    YT.extend(labels)

    train_labels = np.asarray(Y)
    test_labels = np.asarray(YT)

    return train_labels, test_labels


def check_extract_data_features(model):
    if not os.path.exists(model.name + '_features_train.npz') or not os.path.exists(model.name + '_features_test.npz'):
        print("Extracting features for model " + model.name)

        YP = []
        for i in range(1, utils.batch_num + 1):
            print("\tbatch " + str(i))
            data_norm, _ = utils.get_data_from_batch(i)

            prepared_ims = prepare_images(data_norm, model)
            predicted = model.predict(prepared_ims)
            YP.append(predicted)

        YPT = []
        print("\ttest batch")
        data_norm, _ = utils.get_data_from_test_batch()
        prepared_ims = prepare_images(data_norm, model)
        predicted = model.predict(prepared_ims)
        YPT.append(predicted)

        np.savez(model.name + '_features_train', train_features=YP)
        np.savez(model.name + '_features_test', test_features=YPT)
    else:
        print("Loading features for model " + model.name)

    train_features = np.load(model.name + '_features_train.npz')['train_features']
    test_features = np.load(model.name + '_features_test.npz')['test_features']
    train_features = train_features.reshape((-1, model.output_shape[1]))
    test_features = test_features.reshape((-1, model.output_shape[1]))

    return train_features, test_features


def plot_scatter(values, cls, title):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    label_names = utils.get_label_names()
    cmap = cm.rainbow(np.linspace(0.0, 1.0, utils.classes_num))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    for i in range(utils.classes_num):
        cls_idx = np.where(cls == i)[0]
        plt.scatter(x[cls_idx], y[cls_idx], color=colors[cls_idx[0]], label=label_names[i].decode("utf-8"))

    plt.title(title)
    plt.legend()
    plt.show()


def visualizePCA(data, labels):
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)
    plot_scatter(data_reduced, labels, "PCA")


def visualizeTSNE(data, labels):
    # firstly use PCA to reduce number of features
    pca = PCA(n_components=50)
    data_reduced = pca.fit_transform(data)
    tsne = TSNE(n_components=2)
    data_reduced_tsne = tsne.fit_transform(data_reduced)
    plot_scatter(data_reduced_tsne, labels, "t-SNE")


utils.check_download_dataset()
utils.check_extract_dataset()
train_features, test_features = check_extract_inception_features()
train_labels, test_labels = extract_labels()
visualizePCA(train_features, train_labels)
visualizeTSNE(train_features, train_labels)
utils.run_svm(train_features, train_labels, test_features, test_labels)

# here starts the bonus part
# testing different model - InceptionResNet_v2
train_features2, test_features2 = check_extract_inceptionresnet_features()
utils.run_svm(train_features2, train_labels, test_features2, test_labels)
#
# # testing classification on combined set of features extracted by Inception_v3 and InceptionResNet_v2
print("Combining features from Inception v3 and InceptionResNet v2 models")
train_features_combined = np.concatenate((train_features, train_features2), axis=1)
test_features_combined = np.concatenate((test_features, test_features2), axis=1)
utils.run_svm(train_features_combined, train_labels, test_features_combined, test_labels)

# testing data augmentation on Inception_v3 model
augmented_train_features, augmented_train_labels = check_make_data_augmentation()
utils.run_svm(augmented_train_features, augmented_train_labels, test_features, test_labels)
