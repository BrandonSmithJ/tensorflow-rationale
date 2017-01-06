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

Using tensorboard, the overall model and training progress can easily be visualized:

```$ tensorboard --logdir log ```

<img src="https://cloud.githubusercontent.com/assets/1062829/21734584/1ef5ffb6-d432-11e6-91a3-5168fed5c491.png" width=500>
<img src="https://cloud.githubusercontent.com/assets/1062829/21734445/665aa81c-d431-11e6-9c6b-e2eb2118c5c5.png" width=360  align="right">
