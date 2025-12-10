This is an educational experimentation (I am the one being educated here) around the very famous [Attention Is All You Need](https://arxiv.org/pdf/1706.03762). I use a dataset of 130k pairs of translated sentences from English to French to train a machine translation model to translate from French to English, as is described in [this Kaggle competition](https://www.kaggle.com/competitions/neural-machine-translator/).

# Results
Since both French and English languages are Romance-language, I use a common Byte-Pair encoding tokenizer trained on the same dataset. I settled on the following hyperparameters, after some tweaking around. 
```python
num_layers = 4
num_heads = 4
d_model = 128
# Longer sequences (after tokenization) get excluded from the dataset
max_seq_len = 64
vocab_size = 8_000
```

We can probably get much better result with proper hyperparameter tuning but I have absolutely no idea how to do that more systematic way than educated guessing and trying.

On my laptop (with a cheap dedicated GPU from 8 years ago), I get in about an hour a BLEU score of 60% on the Kaggle testing dataset. A sample of the translations:
```txt
> vous n'êtes pas assez agé pour conduire.
you're not old enough to drive.
> nous sommes encore à la maison.
we're still at home.
> je suis heureux de l'entendre.
i'm glad to hear it.
> je suis nul au golf.
i'm not at golf.
> j'ai pris la photo.
i took the picture.
```
# Project Architecture
The file architecture is the following:
- In [attention.py](attention.py) we define a very generic multi-head attention `nn.Module`, that simultaneously handles all the common flavours of attention (cross attention, self attention etc.). The mathematical formulas are very close so it is reasonable to use a single API for all of them.  
- In [transformers.py](transformers.py), we define a standard `TransformerBlock` as defined in the above paper
- In [decoder.py](decoder.py) and [encoder.py](encoder.py) we define two `nn.Module`s following a standard seq2seq architecture, each containing multiple layers of TransformerBlock and doing the appropriate remaining work (embeddings, cross attention in the decoder etc.)
- In [seq2seq.py](seq2seq.py) we define the general translation model and all the training heavylifting is done there: dataloading utilities, the training loop, the testing and generating the final answers.
- In [bpe.py](bpe.py) there's a single BPE tokenizer training and loading function.
- In [my_utils.py](my_utils.py) there are a few PyTorch utilities that are not already existing in PyTorch.  