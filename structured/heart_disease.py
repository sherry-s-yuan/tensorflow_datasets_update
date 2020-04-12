# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Heart Disease Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import tensorflow.compat.v2 as tf

_CITATION = """\
@misc{Dua:2019 ,
author = "Janosi, Steinbrunn and Pfisterer, Detrano",
year = "1988",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml/datasets/Heart+Disease",
institution = "University of California, Irvine, School of Information and Computer Sciences"
}
"""

_DESCRIPTION = """\
This data set contain 13 attributes and labels of heart disease from \
303 participants from Cleveland since Cleveland data was most commonly\
used in modern research.

Attribute by column index
1. #3 (age): age in years
2. #4 (sex): sex (1 = male; 0 = female)
3. #9 (cp): chest pain type
-- Value 1: typical angina
-- Value 2: atypical angina
-- Value 3: non-anginal pain
-- Value 4: asymptomatic
4. #10 (trestbps): resting blood pressure (in mm Hg on admission to the hospital)
5. #12 (chol): serum cholestoral in mg/dl
6. #16 (fbs): (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
7. #19 (restecg): resting electrocardiographic results
8. #32 (thalach): maximum heart rate achieved
9. #38 (exang): exercise induced angina (1 = yes; 0 = no)
10. #40 (oldpeak): ST depression induced by exercise relative to rest
11. #41 (slope): the slope of the peak exercise ST segment
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping
12. #44 (ca): number of major vessels (0-3) colored by flourosopy
13. #51 (thal): 3 = normal; 6 = fixed defect; 7 = reversable defect
14. #58 (num) (the predicted attribute): diagnosis of heart disease (angiographic disease status)
-- Value 0: < 50% diameter narrowing, no presence of heart disease
-- Value 1+: > 50% diameter narrowing, with increasing severity

Dataset Homepage: http://archive.ics.uci.edu/ml/datasets/Heart+Disease
"""


class HeartDisease(tfds.core.GeneratorBasedBuilder):
  """Heart disease dataset with 13 attributes."""

  VERSION = tfds.core.Version("2.0.0", "New split API (https://tensorflow.org/datasets/splits)")
  SUPPORTED_VERSIONS = [
      tfds.core.Version("1.0.0",
                        experiments={tfds.core.Experiment.S3: False}),
  ]

  def _info(self):
    """Define Dataset information"""
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "features": tfds.features.Tensor(shape=(13,), dtype=tf.float32),
            "label": tfds.features.ClassLabel(names=['0', '1', '2', '3', '4'])
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("features", "label"),
        # Homepage of the dataset for documentation
        homepage='http://archive.ics.uci.edu/ml/datasets/Heart+Disease',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    filepath = dl_manager.download('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')
    all_lines = tf.io.gfile.GFile(filepath).read().split("\n")
    records = [l for l in all_lines if ('?' not in l) and l]
    data = [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=1,
            gen_kwargs={"records": records}),
    ]
    return data

  def _generate_examples(self, records):
    """Yields examples."""
    for i, row in enumerate(records):
      elems = row.split(',')
      yield i, {
          "features": [float(e) for e in elems[:-1]],
          "label": elems[-1]
      }
