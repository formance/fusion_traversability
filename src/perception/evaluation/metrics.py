"""Confusion-matrix metrics: rows are ground truth, columns are predictions."""
import numpy as np


def confusion_matrix(truth, prediction, valid, classes=11):
    if truth.shape != prediction.shape or truth.shape != valid.shape:
        raise ValueError('Truth, prediction and mask dimensions must match')
    if valid.dtype != np.bool_:
        raise ValueError('Expected a boolean evaluation mask')
    if truth.dtype.kind not in 'ui' or prediction.dtype.kind not in 'ui':
        raise ValueError('Expected integer class IDs')
    gt, pred = truth[valid], prediction[valid]
    if not len(gt) or np.any(gt < 0) or np.any(gt >= classes) or np.any(pred < 0) or np.any(pred >= classes):
        raise ValueError('Empty evaluation mask or invalid class IDs')
    return np.bincount(classes*gt.astype(np.int64)+pred.astype(np.int64), minlength=classes**2).reshape(classes, classes)


def metrics(matrix, names):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(names), len(names)) or (matrix < 0).any() or not matrix.sum():
        raise ValueError('Expected nonempty nonnegative class confusion matrix')
    gt, pred, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    union = gt+pred-tp
    present = union > 0
    gt_present = gt > 0

    def divide(a, b):
        return float(a/b) if b else None

    per_class = [{'id': i, 'class': name, 'gt_pixels': int(gt[i]), 'predicted_pixels': int(pred[i]),
                  'true_positive': int(tp[i]),
                  'iou': divide(tp[i], union[i]), 'precision': divide(tp[i], pred[i]),
                  'recall': divide(tp[i], gt[i]), 'f1': divide(2*tp[i], gt[i]+pred[i])}
                 for i, name in enumerate(names) if present[i]]
    return {'valid_pixels': int(matrix.sum()), 'pixel_accuracy': float(tp.sum()/matrix.sum()),
            'mean_accuracy_present_gt': float(np.mean(tp[gt_present]/gt[gt_present])),
            'class_ids': np.flatnonzero(present).tolist(),
            'excluded_class_ids': np.flatnonzero(~present).tolist(), 'per_class': per_class}
