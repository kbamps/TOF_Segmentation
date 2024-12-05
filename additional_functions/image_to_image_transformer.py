import os
import json
import random
import uuid
import transforms3d
import numpy as np
import scipy.ndimage
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.utilities import transformations
from tensorflow.keras.utils import to_categorical
from deepvoxnet2.components.transformers import Connection, Transformer, MircInput, SampleInput

class ImageToImageTransformationMatrix(Transformer):
    def __init__(self, subsample_factors = (1,1,1),subsample_factors_ref = (1,1,1),**kwargs):
        super(ImageToImageTransformationMatrix, self).__init__(**kwargs)
        self.subsample_matrix = np.eye(4)
        self.subsample_matrix[:3,:3] = subsample_factors * np.eye(3)
        self.subsample_matrix_ref = np.eye(4)
        self.subsample_matrix_ref[:3, :3] = subsample_factors_ref * np.eye(3)

    def _update_idx(self, idx):
        for idx_, (sample, reference_sample) in enumerate(zip(self.connections[idx][0], self.connections[idx][1])):
            for batch_i in range(len(sample)):
                matrix_A = sample.affine[0, :, :]
                matrix_B = reference_sample.affine[0, :, :]

                matrix_A = np.matmul(matrix_A,self.subsample_matrix)
                matrix_B = np.matmul(matrix_B,self.subsample_matrix_ref)

                image_to_image_matrix = np.matmul(np.linalg.inv(matrix_A), matrix_B)

                transformed_affine = Sample.update_affine(reference_sample.affine, transformation_matrix=None)
                self.outputs[idx][idx_] = Sample(image_to_image_matrix, transformed_affine)

    def _calculate_output_shape_at_idx(self, idx):
        return [(1,4,4,1,1)]

    def _randomize(self):
        pass