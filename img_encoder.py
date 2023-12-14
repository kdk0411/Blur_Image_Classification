import os
import numpy as np
class_encoder = {
    "sharp" : 0,
    "defocused_blurred" : 1,
    "motion_blurred" : 2
}

def img_gather(img_path):
    class_list = os.listdir(img_path)

    file_lists = []
    label_lists = []

    for class_name in class_list:
        file_list = os.listdir(os.path.join(img_path, class_name))
        file_list = list(map(lambda x: "/".join([img_path] + [class_name] + [x]), file_list))
        label_list = [class_encoder[class_name]] * len(file_list)

        file_lists.extend(file_list)
        label_lists.extend(label_list)
    file_lists = np.array(file_lists)
    label_lists = np.array(label_lists)

    return file_lists, label_lists