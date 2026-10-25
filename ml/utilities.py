import numpy as np
import os

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return np.zeros(63)
    else:
        coord_list = []
        hand = results.multi_hand_landmarks[0]
        for i in hand.landmark:
            coord_list.extend([i.x, i.y, i.z])
        return np.array(coord_list)


def normalize_landmarks(coords):
    # TODO: We will build this together step by step!
    pass
