import numpy as np
import torch as th


def get_ids_labels_boxes(box_json):

    num_instances = len(box_json['instances'])

    labels = ids = th.zeros(num_instances, dtype=th.int32)
    boxes = th.zeros((num_instances, 4), dtype=th.int32)

    for i, instance in enumerate(box_json['instances'], 0):
        ids[i] = instance['id']
        labels[i] = instance['cls']
        boxes[i] = th.flatten(th.from_numpy(np.array(instance['roi_rect'])))

    return ids, labels, boxes


def masks_to_boxes(box_json, instance_mask):
    path = 'C:\\Users\\Username\\Path\\To\\File'
    file = open(path, "r")

path = 1


