import numpy as np
import statistics

def compute_segmentation_accuracy_binary(pred, mask):
    element_train_step = mask.size
    correct_train_step = np.equal(pred, mask).sum()

    element_organ_step = np.count_nonzero(mask > 0)
    if element_organ_step > 0:
        coordinates_lung_pixel = np.argwhere(mask > 0)
        correct_organ_step = 0
        for coord in coordinates_lung_pixel:
            if pred[coord[0], coord[1], coord[2]] == mask[coord[0], coord[1], coord[2]]:
                correct_organ_step += 1
        return 0.5*correct_organ_step/element_organ_step + 0.5*(correct_train_step-correct_organ_step)/(element_train_step-element_organ_step)

    else:
        return (correct_train_step)/(element_train_step)

def IoU(pred, mask):
    if np.count_nonzero(mask) > 0:
        # worm prediction IoU score
        num = np.sum(pred*mask)
        denum = np.sum(pred + mask - (pred*mask))
        worm_score = num/denum

        #background prediction IoU score
        pred = (1-pred)
        mask = (1-mask)
        num = np.sum(pred*mask)
        denum = np.sum(pred + mask - (pred*mask))
        background_score = num/denum
        return (worm_score + background_score)/2
    else :
        #background prediction IoU score
        pred = (1-pred)
        mask = (1-mask)
        num = np.sum(pred*mask)
        denum = np.sum(pred + mask - (pred*mask))
        background_score = num/denum
        return background_score