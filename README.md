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


### Performance

<img src="https://cloud.githubusercontent.com/assets/1062829/22362597/bb73bf2c-e431-11e6-804c-ccf8c17c3965.png" align="right", width=550>

Initially the generator randomly selects text, which appears as a uniform noise in the visualization. As the encoder provides feedback for the selections, the generator begins creating a more sparse representation, eventually converging to groups of words in each text sample. The images below show different points in the training process, with the latter showing groups of rationals emerging (where the vertical axis represents the text document, and the horizontal represents the batch dimension). The dark blue portions of the images represent padding, which is ignored by the model.

The model as constructed tends to bounce between a small sampling rate and a large one, primarily due to adding a higher cost to the evaluation function at both ends (too little text and too much). This has the effect of allowing the model to recheck previously discarded text later in training, thereby giving a chance to re-evaluate in the context of better learned weights. This also causes a slower overall convergence, but in essence performs a regularization on selections. Alternative cost functions may speed up the convergence to text segments by penalizing small chains of text higher than longer ones, in a more explicit manner than currently implemented.
<br>
<br>

<br>
Perfect convergence (on the current dataset) is likely impossible, due to the subjectivity between scores and reviews. Take for example the following (test set) sample:
<br>
<img src="https://cloud.githubusercontent.com/assets/1062829/22362673/72d31a1e-e432-11e6-8d3c-256e35002a8d.png">
<br>
<br>
It would be difficult to rationalize the prediction even as a human, due to the low information density and inherent ambiguity.
As well, outliers tend to be difficult for the network to predict:
<br>
<img src="https://cloud.githubusercontent.com/assets/1062829/22362705/af72517e-e432-11e6-9936-17652821d704.png">
<br>
<br>
For many however, the network performs well both on the prediction and the text selection to justify that prediction:
<br>
<img src="https://cloud.githubusercontent.com/assets/1062829/22362712/b9a12fb2-e432-11e6-8c5e-068bcdf775e8.png">
<br>
<img src="https://cloud.githubusercontent.com/assets/1062829/22362724/c531c0f8-e432-11e6-97cf-3e17a96cf691.png">
<br>
<br>
One thing to note: though these samples are predicted on the given rating for 'taste', the network doesn't necessarily select text which deals with taste directly (e.g. specific flavors). Instead, it selects the text which is most predictive of the score given; this text is entirely dependent upon what the sample writers thought contributed to score they gave.
