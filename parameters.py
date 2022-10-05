import torch

GRAPH_PAR=[7,7,7,7,7]
SEB_NUM=5
HID_DIM=100

TRAIN_SIZE = 1500
SEQ_LEN=112
SEQ2_LEN=100
EPOCH=600
LR1 = 0.0012
LR2 = 0.0005
BATCH_SIZE = 30

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
