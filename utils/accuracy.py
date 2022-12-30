import numpy as np

def compute_segmentation_accuracy_binary(pred: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates the segmentation accuracy for a binary classification problem.
    
    Parameters:
        pred (numpy.ndarray): The predicted labels.
        ground_truth (numpy.ndarray): The ground truth labels.
        
    Returns:
        float: The segmentation accuracy.
    """
    
    # Calculate the number of pixels in the mask and the number of correctly classified pixels
    ground_truth_size = ground_truth.size
    number_of_correct_pixels = np.equal(pred, ground_truth).sum()

    # Calculate the number of labeled pixels in the ground truth mask
    number_of_labeled_pixels = np.count_nonzero(ground_truth > 0)
    
    # If there are no labeled pixels in the ground truth mask, return the global accuracy
    if number_of_labeled_pixels == 0:
        return number_of_correct_pixels / ground_truth_size
    
    # Otherwise, calculate the accuracy of labeled pixels and the accuracy of background pixels separately
    number_of_correctly_labeled_pixels = 0
    for coord in np.argwhere(ground_truth > 0):
        if pred[tuple(coord)] == ground_truth[tuple(coord)]:
            number_of_correctly_labeled_pixels += 1
    return (0.5 * number_of_correctly_labeled_pixels / number_of_labeled_pixels) + 0.5 * (number_of_correct_pixels - number_of_correctly_labeled_pixels) / (ground_truth_size - number_of_labeled_pixels)


def IoU(pred: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Computes the Intersection over Union (IoU) score for binary segmentation.
    
    Parameters:
        pred (numpy.ndarray): The predicted mask.
        ground_truth (numpy.ndarray): The ground truth mask.
        
    Returns:
        float: The IoU score.
    """
    if pred.shape != ground_truth.shape:
        raise ValueError('Inputs must have the same shape')

    # Convert both arrays to int in case they were bools
    pred = pred.astype(int)
    ground_truth = ground_truth.astype(int)

    # If there are no labeled pixels in the ground truth mask, just compute IoU score for the background
    if np.count_nonzero(ground_truth) == 0:
        pred = (1-pred)
        ground_truth = (1-ground_truth)
        num = np.sum(pred*ground_truth)
        denum = np.sum(pred + ground_truth - (pred*ground_truth))
        background_score = num/denum
        return background_score

    # Else, return the mean of the background and labeled pixels IoU scores
    else:
        # labeled pixels IoU score
        num = np.sum(pred*ground_truth)
        denum = np.sum(pred + ground_truth - (pred*ground_truth))
        worm_score = num/denum

        # background prediction IoU score
        pred = (1-pred)
        ground_truth = (1-ground_truth)
        num = np.sum(pred*ground_truth)
        denum = np.sum(pred + ground_truth - (pred*ground_truth))
        background_score = num/denum
        return (worm_score + background_score)/2