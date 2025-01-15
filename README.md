Repository for MSC project on Polyphonic DDSP for Guitar timbre transfer. Includes implementation of DDSP network used for  guitar timbre transfer

requirements.txt contains the details on package versions required to run the code.

PolyDDSP folder is a snapshot of the work in progress repository for the proposed PolyDDSP system. Some minor modifications were made to the input of the noise synthesizer to account for a variation in the form of the input data. This directory is not my original work. A link to the repository follows: https://github.com/TeeJayBaker/PolyDDSP/tree/main

The gtt folder contains all code that was written for the project.

dataloader.py contains the feature extractor and dataset objects used to augment the training data and retrieve the audio features.

model.py contains the gtt network used to synthesize audio.

decoder.py contains the decoder architecture used to synthesizer DDSP synthesizer controls.

eval.py contains the support functions used to test the model as well as the evaluation functions.

train.py contains functions related to the training loop.​

timbre.timbre_encoder.py contains the code used to construct the layers of the timbre encoder.

utilities.utils.py contains an assortment of general helper functions called at various points in the repository.

Experiments.ipynb contains the code used to run the experiments described in the paper

Visualizations and Examples.ipynb contains the code used to generate the audio examples and figures

decoder_old.py and model_old.py contain early versions of the model configuration. These variants were not referenced in the paper, but were considered during an earlier stages of development. They resemble the design used in the snapshot of the PolyDDSP repository included in the repository. It should be noted that the design in the snapshot differs from the design described in the paper and bears similarity to the monophonic DDSP network, but for the polyphonic case.




