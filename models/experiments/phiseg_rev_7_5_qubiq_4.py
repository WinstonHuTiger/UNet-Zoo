import torch
import torch.nn as nn
from models.phiseg import PHISeg
from utils import normalise_image
from data.qubiq_data import qubiq_data

experiment_name = 'PHISegRev_7_5_qubiq_4_brain_growth'
log_dir_name = 'qubiq'

data_loader = qubiq_data 
dataset =  "brain-growth"
output = "annotator"
data_root = r'D:\dev_x\bAttenUnet_test\qubiq'
task = 0

# number of filter for the latent levels, they will be applied in the order as loaded into the list
filter_channels = [32, 64, 128, 192, 192, 192, 192]
latent_levels = 5

iterations = 5000000

n_classes = 2
num_labels_per_subject = 7

no_convs_fcomb = 4 # not used
beta = 10.0 # not used
#
use_reversible = True
exponential_weighting = True

# use 1 for grayscale, 3 for RGB images
input_channels = 1
epochs_to_train = 20
batch_size = 4
image_size = (1, 256, 256)

augmentation_options = {'do_flip_lr': False,
                        'do_flip_ud': False,
                        'do_rotations': False,
                        'do_scaleaug': False,
                        'nlabels': n_classes}

input_normalisation = normalise_image

validation_samples = 1
num_validation_images = 100

logging_frequency = 1000
validation_frequency = 1000

weight_decay = 10e-5

pretrained_model = None
# model
model = PHISeg