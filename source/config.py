EMOTION_DECLARATION = [
    "neutral",
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "sadness",
    "surprise",
]

TEST_SPLIT = 0.1
TRAIN_SPLIT = 0.8 * (1 / (1-TEST_SPLIT))
VAL_SPLIT = 0.1 * (1/(1-TEST_SPLIT))

GOOGLE_TRAIN_SPLIT = 1
CK_TRAIN_SPLIT = 1

GOOGLE_TEST_SPLIT = 0
CK_TEST_SPLIT = 1

IMG_SIZE = 224
MIN_PIC_SIZE = 20  # MIN FACE SIZE
