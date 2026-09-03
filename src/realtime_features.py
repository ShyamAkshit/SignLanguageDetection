# =========================================
# REAL-TIME FEATURE PREPARATION
# =========================================


def prepare_realtime_features(features):

    # -----------------------------------------
    # Check original feature count
    # -----------------------------------------

    if len(features) != 63:
        raise ValueError(
            f"Expected 63 features, but received {len(features)}"
        )


    # -----------------------------------------
    # Remove constant features
    #
    # f0, f1, f2
    # -----------------------------------------

    processed_features = features[3:]


    # -----------------------------------------
    # Check final feature count
    # -----------------------------------------

    if len(processed_features) != 60:
        raise ValueError(
            f"Expected 60 features after preprocessing, "
            f"but received {len(processed_features)}"
        )


    return processed_features


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    print("----------------------------------")
    print("Real-Time Feature Preparation")
    print("----------------------------------")


    # Create a test 63-feature vector

    test_features = list(range(63))


    print("\nOriginal feature count:")
    print(len(test_features))


    # Prepare features

    processed_features = prepare_realtime_features(
        test_features
    )


    print("\nProcessed feature count:")
    print(len(processed_features))


    print("\nFirst 5 processed features:")
    print(processed_features[:5])


    print("\nLast 5 processed features:")
    print(processed_features[-5:])


    print("\n----------------------------------")
    print("Feature Preparation Test Complete")
    print("----------------------------------")