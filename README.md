This is an educational experimentation (I am the one being educated here) around the very famous [Attention Is All You Need](https://arxiv.org/pdf/1706.03762). I use a dataset of 130k pairs of translated sentences from English to French to train a machine translation model to translate from French to English. 

# Results
On my very small laptop (with a cheap dedicated GPU from 8 years ago), I get 
TODO train and show results

# Architecture
The file architecture is the following:
- In [attention.py](attention.py) we define a very generic multi-head attention `nn.Module`, that simultaneously handles all the common flavours of attention (cross attention, self attention etc.). The mathematical formulas are very close so it is reasonable to use a single API for all of them.  
- In [transformers.py](transformers.py), we define a standard `TransformerBlock` as defined in the above paper
- In [decoder.py](decoder.py) and [encoder.py](encoder.py) we define two `nn.Module`s following a standard seq2seq architecture, each containing multiple layers of TransformerBlock and doing the appropriate remaining work (embeddings, cross attention in the decoder etc.)
- In [seq2seq.py](seq2seq.py) we define the general translation model and all the training heavylifting is done there: dataloading utilities, the training loop, the testing etc.
