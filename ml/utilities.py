import numpy as np

def extract_landmarks(result):
    landmarks = getattr(result, "hand_landmarks", None) or getattr(result, "multi_hand_landmarks", None)
    if not landmarks or len(landmarks) == 0:
        return np.zeros(63)
    else:
        hand = landmarks[0]
        joint_list = getattr(hand, "landmark", hand)
        coord_list = []
        for joint in joint_list:
            coord_list.extend([joint.x, joint.y, joint.z])
        return np.array(coord_list)

def normalize_landmarks(coords: np.ndarray):
    coord_reshaped = coords.reshape(21,3)
    wrist = coord_reshaped[0]
    normalized_array = (coord_reshaped - wrist).reshape(63,)
    max_val = np.max(np.abs(normalized_array))
    if max_val > 0:
        normalized_array = normalized_array / max_val
    return normalized_array
