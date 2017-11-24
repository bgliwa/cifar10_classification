import matplotlib.pyplot as plt
import numpy as np
import utils


def visualize_random_images_in_classes():
    #we use data from 1st batch for visualization
    data_norm, labels = utils.get_data_from_batch(1)
    label_names = utils.get_label_names()
    classes_num = 10
    samples_num = 10
    labels_nd = np.array(labels)
    fig, axes = plt.subplots(classes_num, samples_num + 1, figsize=(9, 9))
    for i in range(classes_num):
        axes[i][0].set_axis_off()
        axes[i][0].text(-0.5, 0.5, label_names[i].decode("utf-8"))
        indexes = np.where(labels_nd == i)[0]
        for j in range(1, samples_num + 1):
            k = np.random.choice(indexes, replace=False)
            axes[i][j].set_axis_off()
            axes[i][j].imshow(data_norm[k])
    plt.show()


utils.check_download_dataset()
utils.check_extract_dataset()
visualize_random_images_in_classes()
