### Understanding Prediction via Unsupervised Rationale Generation

This repo contains a tensorflow implementation of the system described in <br>
<b>"Rationalizing Neural Predictions". Tao Lei, Regina Barzilay and Tommi Jaakkola. EMNLP 2016.  [[PDF]](https://people.csail.mit.edu/taolei/papers/emnlp16_rationale.pdf)</b>

A theano implementation is available from: https://github.com/taolei87/rcnn


To run the code with default parameters:
```
$ python3 model.py --embedding=data/review+wiki.filtered.200.txt.gz --training=data/reviews.aspect1.train.txt.gz --testing=data/reviews.aspect1.heldout.txt.gz --output=output.json
```

Data sets can be found through the github linked above. Embeddings may be any word vector sets, such as the ones found [here](http://nlp.stanford.edu/projects/glove/).

<br>

### Overview

The objective is to create a model which can simultaneously classify text documents, and provide justifications for those classifications via the text itself. By specifying two sub-components - a Generator and an Encoder - and training them in concert, the model learns to choose concise phrases which are then used to make the classification. 

<img src="https://cloud.githubusercontent.com/assets/1062829/21734445/665aa81c-d431-11e6-9c6b-e2eb2118c5c5.png" width=360  align="right">
Using tensorboard, the model can easily be visualized. The two components are linked at each step, such that they form a step-ladder reinforcement scheme. 

In other words, the generator determines which text must be selected by gradually increasing or decreasing the probability of selection based on the encoder's ability to predict a classification on the selection. In turn, the encoder learns to predict the correct classification for a given text based on the snippets of text given by the generator. 

This reinforcement pattern must be balanced to acheive learning - too much weight on either component overwhelms the other and the model converges suboptimally. 


Initially the generator randomly selects text, which appears as a uniform noise in the visualization. As the encoder provides feedback for the selections, the generator begins creating a more sparse representation, eventually converging to groups of words in each text sample. The two images below show different points in the training process, with the latter beginning to show groups of rationals emerging (where the vertical axis represents the text document, and the horizontal represents the batch dimension). 

<img src="https://cloud.githubusercontent.com/assets/1062829/21827821/48be153c-d75b-11e6-955a-8869cd89e123.png">




