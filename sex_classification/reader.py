from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np

from dltk.io.augmentation import extract_random_example_array, flip
from dltk.io.preprocessing import whitening


def read_fn(file_references, mode, params=None):
    #augments the function
    def _augment(img):
        return flip(img, axis=2)

    for f in file_references:
        subject_id = f[0]

        data_path = '../Data/IXI_HH/2mm'

        # Read the image nii with sitk
        t1_fn = os.path.join(data_path, '{}/T1_2mm.nii.gz'.format(subject_id))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(t1_fn)))

        # Normalise the image using dltk library function
        # Makes computation efficient by reducing values to between 0 and 1

        t1 = whitening(t1)

        images = np.expand_dims(t1, axis=-1).astype(np.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}, 'img_id': subject_id}

        # sex is represented by 1's and 2's, here we shift them to 0's and 1's
        sex = np.int(f[1]) - 1
        y = np.expand_dims(sex, axis=-1).astype(np.int32)

        # Augment the data for training only
        if mode == tf.estimator.ModeKeys.TRAIN:
            images = _augment(images)

        # If we are deploying we don't need to extract every image since that was already done during training
        if params['extract_examples']:
            images = extract_random_example_array(
                image_list=images,
                example_size=params['example_size'],
                n_examples=params['n_examples'])

            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': y.astype(np.float32)},
                       'img_id': subject_id}

        else:
            yield {'features': {'x': images},
                   'labels': {'y': y.astype(np.float32)},
                   'img_id': subject_id}
        # yields a dictionary of reader outputs for dltk.io.abstract_reader

    return
