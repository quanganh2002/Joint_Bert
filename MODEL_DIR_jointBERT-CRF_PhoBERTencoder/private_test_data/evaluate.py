# NOTE: we only support python2.7 on AIHUB.VN at the moment,
# be mindful to check what is supported in sci-kit learns of python2.7 (scikitlearn < 0.22.0)
import enum
import os
import sys
import numpy as np

import csv


def get_sentence_frame_acc(intent_label, intent_pred, slot_label, slot_pred):
    """For the cases that intent and all the slots are correct (in one sentence)
    
    ::params::
    intent_label: list containing intent labels
    intent_pred: list containing intent predictions
    slot_label: list containing lists of slot labels
    slot_pred: list containing lists of slot predictions
    """
    try:
        assert len(intent_label) == len(intent_pred) and len(slot_label) == len(slot_pred)
        
        # Get the intent comparison result
        intent_result = [i == j for i, j in zip(intent_label, intent_pred)]

        # Get the slot comparision result
        slot_result = []
        count = 0
        for idx, (preds, labels) in enumerate(zip(slot_pred, slot_label)):
            if len(preds) != len(labels):
                print(idx)
                # raise Exception(f"Number of slots predicted is not equal to number of syllables. Sentence number {idx}")
            one_sent_result = True
            for p, l in zip(preds, labels):
                if p != l:
                    one_sent_result = False
                    break
            slot_result.append(one_sent_result)
            count += 1
        slot_result = np.array(slot_result)

        semantic_acc = np.multiply(intent_result, slot_result).mean()        
        return semantic_acc

    except AssertionError:
        raise Exception("Number of predictions is not equal to number of labels")


if __name__ == "__main__":
    # read ground truth
    intent_label, slot_label = [], []
    with open('private_ground_truth.csv') as ground_truth_file:
        csv_reader = csv.reader(ground_truth_file, delimiter=',')
        for row in csv_reader:
            intent_label.append(row[0])
            slot_label.append(row[1].strip().split())

    # read predictions
    intent_pred = []
    slot_pred = []
    with open("submission_results.csv") as results_file:
        csv_reader = csv.reader(results_file, delimiter=',')
        for row in csv_reader:
            intent_pred.append(row[0])
            slot_pred.append(row[1].strip().split())
    
    # calculate metrics
    sentence_acc = get_sentence_frame_acc(intent_label, intent_pred, slot_label, slot_pred)

    with open('scores.txt', 'w') as output_file:
        output_file.write("Sentence accuracy: {:f}\n".format(round(sentence_acc, 4)))
