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
* Eventually there will be two most important files in nanogpt
1. `model.py` GPT model definition defines the transformer model
2. `train.py` trains on the given text such as Shakespeare, Transcripts or any other text or text like corpus such as Windows file names.

* But, as I build this code, there are `bigram.py` file to illustrate how a bigram model which is easy to understand can potentially be translated into a neural network and we can incorporate all pieces into it.
* However, the bigram model doesn't allow the precending tokens from length `block_size` previous tokens to talk to the current token because by definition, bigram model only looks at one previous token and itself.
* For validation and comparisons with the OpenAI weights, we can optionally load the GPT-2 weights from OpenAI and validate our model weights are equivalent to theirs.

## Mathematical tricks for self attention
* `mathtrick.py` file contains examples for the mathematical trick used to implement the self attention which is getting the tokens (characters for this implementation) to talk to each other. 
* When the tokens can talk to each other for the same 'x' of shape B, T, C (Batch, Timesteps, Channels), then it is called the self attention or decoder only model.
* In contrast, when the tokens in Q,K,V tensors can talk outside of the same 'x' on the side, such as K,V of machine translation can get those tensors from a pre-trained encoder which has been trained on French language and on the side, the Key and Value tensors are then utilized by the decoder model in order to dot product with the Query vector.
* The above method can be used to train any sequence on a context that is provided by a separate encoder such as healthcare codes can be used as vocabulary and encoded to payment and transaction outcomes. 

