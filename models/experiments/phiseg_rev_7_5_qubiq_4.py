import torch
import torch.nn as nn
from models.phiseg import PHISeg
from utils import normalise_image
from data.qubiq_data import qubiq_data

experiment_name = 'PHISegRev_7_5_qubiq_4_brain_tumor'
log_dir_name = 'qubiq'

data_loader = qubiq_data 
dataset =  "kidney"
output = "annotator"

# number of filter for the latent levels, they will be applied in the order as loaded into the list
filter_channels = [32, 64, 128, 192, 192, 192, 192]
latent_levels = 5

iterations = 5000000

n_classes = 2
num_labels_per_subject = 4

no_convs_fcomb = 4 # not used
beta = 10.0 # not used
#
use_reversible = True
exponential_weighting = True

# use 1 for grayscale, 3 for RGB images
input_channels = 1
epochs_to_train = 20
batch_size = 12
image_size = (4, 256, 256)

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': n_classes}

input_normalisation = normalise_image

validation_samples = 4
num_validation_images = 100

logging_frequency = 1000
validation_frequency = 1000

weight_decay = 10e-5

pretrained_model = None
# model
model = PHISeg