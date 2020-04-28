#Import Libraries
from fastai.vision import *
from fastai.metrics import error_rate

#Define Path for dataset
path = untar_data('http://files.fast.ai/data/examples/mnist_sample')


#Creating the databunch
data = ImageDataBunch.from_folder(path, ds_tfms = (rand_pad(2, 28), []), bs = 64)
data.normalize(imagenet_stats)

#Create the model and learn
learn = cnn_learner(data, models.resnet18, metrics = accuracy)
learn.fit_one_cycle(1, 0.01)

#View the output
print("Prediction Accuracy = ", accuracy(*learn.get_preds()))

