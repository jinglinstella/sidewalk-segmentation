Collaboration work by: 
Jinglin Tao (jt4296)
Grace Chen (fc2286)

Summary:

  Overall framework: SamGeo

  Reference: https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb

  Implementation Enviroment: Google Colab

  Source Data: Sidewalk train and validation data streamed from Huggingface

  Training: SGD with DiceFocalLoss loss function

  Two sets of Training are performed:

    1) 10 epochs training on a subset of train data (8000 images)
    2) 2 epochs training on a full set of train data

  Inference:

    a) Prediction on a randomly selected tif image from train dataset based on model parameters trained from 1)

    b) Prediction on a randomly selected tif image from validation dataset based on model parameters trained from 1)

    c) Prediction on a randomly selected tif image from train dataset based on model parameters trained from 2)

    d) Prediction on a randomly selected tif image from validation dataset based on model parameters trained from 2)

  

Findings:

The prediction outcome was much better from 10 epoches training on a subset of train data (with sidewalk more clearly being segmented). 

However this outcome may vary should different optimizer and hypeparameters being chosen

Understand that in theory if more training data are used, the generalization would be better



Follow up work:

1) Test different optimizer and hypeparameters for training
2) more epochs training to be achieve should time allow

