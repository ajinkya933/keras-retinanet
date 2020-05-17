from keras import backend as K
import tensorflow as tf

from tensorflow.python import saved_model
from tensorflow.python.saved_model.signature_def_utils_impl import (
    build_signature_def, predict_signature_def
)

import shutil
import os
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.gpu import setup_gpu
from ..utils.keras_version import check_keras_version
from ..utils.tf_version import check_tf_version



export_path = '/home/ajinkya/Documents/ashnair1/keras-retinanet/retinanet_reducedanchor_savedmodel'
weight_file_path = '/home/ajinkya/Documents/ashnair1/keras-retinanet/snapshots_reduced_anchorsize/reduced_anchorsize_snapshots_resnet50_csv_50.h5'
h5_model = models.load_model(weight_file_path)

# optionally load config parameters
anchor_parameters = None
#if args.config:
config_filename = '/home/ajinkya/Documents/ashnair1/keras-retinanet/config.ini'
config = read_config_file(config_filename)
if 'anchor_parameters' in config:
    anchor_parameters = parse_anchor_parameters(config)


model = models.convert_model(
    h5_model,
    nms=True,
    class_specific_filter=False,
    anchor_params=anchor_parameters
)
#model.load_weights('/home/ajinkya/Documents/ashnair1/keras-retinanet/snapshots_reduced_anchorsize/reduced_anchorsize_snapshots_resnet50_csv_50.h5')

print('Output layers', [o.name[:-2] for o in model.outputs])
print('Input layer', model.inputs[0].name[:-2])

# Output layers: ['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3', 'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3', 'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3']
# Input layer: input_1

if os.path.isdir(export_path):
    shutil.rmtree(export_path)
builder = saved_model.builder.SavedModelBuilder(export_path)

signature = predict_signature_def(
    inputs={'images': model.input},
    outputs={
        'output1': model.outputs[0],
        'output2': model.outputs[1],
        'output3': model.outputs[2]
    }
)

sess = K.get_session()
builder.add_meta_graph_and_variables(sess=sess,
                                     tags=[saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict': signature})
builder.save()

