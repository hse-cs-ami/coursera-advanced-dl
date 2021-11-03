from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as t
from torchvision.ops import nms


def torch_to_pil(img):
    return t.ToPILImage()(img).convert('RGB')


def plot_img_bbox(img, target: dict, prediction: dict = None):
    img = torch_to_pil(img)

    target = {k: v.cpu() for k, v in target.items()}
    prediction = {k: v.cpu() for k, v in prediction.items()} if prediction is not None else None

    if prediction is not None:
        prediction = apply_nms(prediction, iou_thresh=0.3)

        fig, a = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)

        a[0].axis('off')
        a[1].axis('off')

        a[0].imshow(img)
        a[1].imshow(img)

        a[0].set_title(f'Target, label = [{target["labels"].item()}]')

        for box in target['boxes']:
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle(
                (x, y),
                width, height,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            a[0].add_patch(rect)

        if len(prediction['labels']):
            a[1].set_title(f'Prediction, label = [{prediction["labels"][0].item()}]')

            for box in prediction['boxes']:
                x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
                rect = patches.Rectangle(
                    (x, y),
                    width, height,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                a[1].add_patch(rect)
        else:
            a[1].set_title(f'Prediction empty')
    else:
        fig, a = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)

        a.axis('off')

        a.imshow(img)

        a.set_title(f'Target, label = [{target["labels"].item()}]')

        for box in target['boxes']:
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle(
                (x, y),
                width, height,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            a.add_patch(rect)

    plt.show()


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union


def calculate_map(
        det_boxes: List[torch.Tensor], det_labels: List[torch.Tensor], det_scores: List[torch.Tensor],
        true_boxes: List[torch.Tensor], true_labels: List[torch.Tensor], device: torch.device
):
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)

    n_classes = 201

    true_images: list = []

    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images: torch.Tensor = torch.LongTensor(true_images).to(device)

    true_boxes: torch.Tensor = torch.cat(true_boxes, dim=0)
    true_labels: torch.Tensor = torch.cat(true_labels, dim=0)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    det_images: list = []
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images: torch.Tensor = torch.LongTensor(det_images).to(device)

    det_boxes: torch.Tensor = torch.cat(det_boxes, dim=0)
    det_labels: torch.Tensor = torch.cat(det_labels, dim=0)
    det_scores: torch.Tensor = torch.cat(det_scores, dim=0)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    average_precisions: torch.Tensor = torch.zeros((n_classes - 1), dtype=torch.float)

    for c in range(1, n_classes):
        true_class_images = true_images[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]
        n_class_objects = true_class_boxes.size(0)  # ignore difficult objects

        true_class_boxes_detected = torch.zeros(n_class_objects, dtype=torch.uint8).to(device)

        det_class_images = det_images[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        n_class_detections = det_class_boxes.size(0)

        if n_class_detections == 0:
            continue

        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]

        true_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)
        false_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)
            this_image = det_class_images[d]

            object_boxes = true_class_boxes[true_class_images == this_image]

            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)

            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]

            if max_overlap.item() > 0.5:
                if true_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_ind] = 1
                else:
                    false_positives[d] = 1
            else:
                false_positives[d] = 1

        cum_true_positives = torch.cumsum(true_positives, dim=0)
        cum_false_positives = torch.cumsum(false_positives, dim=0)
        cum_precision = cum_true_positives / (cum_true_positives + cum_false_positives + 1e-10)
        cum_recall = cum_true_positives / n_class_objects

        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)

        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cum_recall >= t
            if recalls_above_t.any():
                precisions[i] = cum_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()

    mean_average_precision = average_precisions.mean().item()

    return mean_average_precision


def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction
