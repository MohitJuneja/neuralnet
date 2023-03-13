# neuralnet
Autograd and Neural Network from scratch


## Landmark Papers
https://arxiv.org/abs/1706.03762 Attention is all you need
1. GPT : Generatively Pretrained Transformers
2. What: Arch. proposed for tasks such as machine translation
3. Why: Writers had no idea that this paper will take the AI by storm
4. What Next: The transformer architecture took the whole world and rest of AI by storm.


#### The architecture was copied pasted into huge amount of application. 

ChatGPT is trained ona good chunk of Internet and is a production grade system trained on possibly these datasets listed in a documentation of an OSS version of "ChatGPT like" system: https://projects.laion.ai/Open-Assistant/docs/data/supervised-datasets

## Basics
* Character level language model will be educational on how these systems work.
* Learn from a smaller dataset such as a sample https://raw.githubusercontent.com/borkabrak/markov/Complete-Works-of-William-Shakespeare.txt
* Given characters of Shakespeare, we will generate fake Shakespeare from Transformer such as ChatGPT

## This code comes from nanoGPT - credits to Andrej Karpathy
* Repository for simple way to train transformers.
* Two Files in nanogpt
1. `model.py` GPT model definition defines the transformer model
2. `train.py` trains on the given text such as Shakespeare, Transcripts or any other text or text like corpus such as Windows file names.

* For validation and comparisons with the OpenAI weights, we can optionally load the GPT-2 weights from OpenAI and validate our model weights are equivalent to theirs.


