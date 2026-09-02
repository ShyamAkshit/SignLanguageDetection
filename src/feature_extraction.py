import math


def extract_landmarks(hand_landmarks):

    # -----------------------------------------
    # Step 1: Get wrist landmark
    # -----------------------------------------

    wrist = hand_landmarks[0]

    relative_landmarks = []

    # -----------------------------------------
    # Step 2: Make all landmarks relative
    # to the wrist
    # -----------------------------------------

    for landmark in hand_landmarks:

        relative_x = landmark.x - wrist.x
        relative_y = landmark.y - wrist.y
        relative_z = landmark.z - wrist.z

        relative_landmarks.append(
            (relative_x, relative_y, relative_z)
        )

    # -----------------------------------------
    # Step 3: Calculate hand scale
    # using Euclidean distance
    # -----------------------------------------

    scale = 0

    for x, y, z in relative_landmarks:

        distance = math.sqrt(
            x * x +
            y * y +
            z * z
        )

        scale = max(scale, distance)

    # -----------------------------------------
    # Step 4: Avoid division by zero
    # -----------------------------------------

    if scale == 0:
        scale = 1

    # -----------------------------------------
    # Step 5: Normalize
    # -----------------------------------------

    features = []

    for x, y, z in relative_landmarks:

        normalized_x = x / scale
        normalized_y = y / scale
        normalized_z = z / scale

        features.extend([
            normalized_x,
            normalized_y,
            normalized_z
        ])

    return features