DEVICE = "cpu"  # if you have GPU; otherwise "cpu"

DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints"

MODEL_CHECKPOINT = f"{CHECKPOINT_DIR}/cifar10_resnet18.pt"

TARGET_CLASS = 3

PATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 12
STEPS_PER_EPOCH = 400
TV_LAMBDA = 0.001   # Optional, improves visual quality


