"""Basic experiment configuration

For different models, a specific configuration can be created based on this basic setting.

"""

from yacs.config import CfgNode as CN

_C = CN()

# public alias
cfg = _C

# -----------------------------------------
# Model
# -----------------------------------------
_C.MODEL = CN()
# Overwritten by different tasks
_C.MODEL.TYPE = ""
# Pre-trained weights
_C.MODEL.WEIGHT = ""
_C.MODEL.EMBEDDING_SIZE = 64
_C.MODEL.REGS = 1e-5
# Regulation term for GAN
_C.MODEL.REGSD = 1e-5
_C.MODEL.REGSG = 1e-5

_C.MODEL.MARGIN = 1.0

# -----------------------------------------
# Dataset
# -----------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = ""
# Root directory for dataset
_C.DATASET.ROOT_DIR = ""
# Sampling strategies
_C.DATASET.SAMPLING = "RANDOM"
# Number of negative items for each positive item
_C.DATASET.K_NEGS = 1
_C.DATASET.NUM_STEP = 3

# -----------------------------------------
# Dataloader
# -----------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 1

# -----------------------------------------
# Sovler(optimizer)
# -----------------------------------------
_C.SOLVER = CN()
# Type of optimizer
_C.SOLVER.TYPE = "Adam"

# Basic parameters for solvers
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WEIGHT_DECAY = 0.0

# Setting for GAN
_C.SOLVER.TYPEG = "Adam"

# Basic parameters for solvers
_C.SOLVER.BASE_LRG = 0.001
_C.SOLVER.WEIGHT_DECAYG = 0.0

_C.SOLVER.TYPED = "Adam"

# Basic parameters for solvers
_C.SOLVER.BASE_LRD = 0.001
_C.SOLVER.WEIGHT_DECAYD = 0.0


# -----------------------------------------
# Train
# -----------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1024
_C.TRAIN.LOG_PERIOD = 10
_C.TRAIN.MAX_EPOCH = 400
_C.TRAIN.EARLY_STOP = 3
_C.TRAIN.TYPE = ""

# -----------------------------------------
# Test
# -----------------------------------------
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1024
_C.TEST.KS = [20, 40, 60, 80, 100]
# metric to evaluate model for early stop
# Four metrics are available:
# - RECALL - PRECISION - NDCG - HIT_RATIO
_C.TEST.METRIC = "RECALL"

_C.TEST.NUM_BATCH = 4
# -----------------------------------------
# Dataset
# -----------------------------------------
# if set to @, use default filename
_C.OUTPUT_DIR = "@"
# For reproducibility... but not really, because modern fast
# GPU libraries use non-deterministic op implementation
_C.RANDOM_SEED = 2019
