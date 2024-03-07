# Min-Translator: A Minimalist Machine Translation Network From Scratch

Welcome to Min-Translator, a repository dedicated to providing a slightly more than minimum-viable machine translation networks for educational purposes. The goal is to help you understand the inner workings of these networks. 

## Technologies Used

The architectures are built using the [PyTorch](https://pytorch.org) library on Python 3.11 and are trained on a [French-to-English dataset](https://www.manythings.org/anki/). A basic word-level tokenizer from scratch has also been implemented using `numpy`. 

**Please note:** While this tokenizer gets the job done, it's not comparable to tools such as [`torchtext`](https://pytorch.org/text/stable/index.html) and [Hugging Face Tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer). This repository is not intended to serve as a tutorial for building a tokenizer.

## Architectures

The following architectures are implemented from these papers:

1. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Basic Seq2Seq)
2. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Seq2Seq with Bahdanau (Additive) Attention)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Encoder-Decoder Transformer)

## Why Min-Translator?

Many projects that implement these machine translation networks from scratch often have uncommented or unreadable code. Some that implement the [Transformer](https://arxiv.org/abs/1706.03762) often implement the Encoder or Decoder, but not both. Min-Translator aims to fix these issues, providing a clear path to understanding the Transformer and the concepts it introduces and builds upon.

## Getting Started

This repository should be run in Python 3.11. The required dependencies are provided in `requirements.txt` and can be installed via `pip install -r requirements.txt`. It is recommended to run this within a [virtual environment](https://docs.python.org/3/library/venv.html).

## TODO

This repository is still in development. The following tasks still need to be completed:
- Proper documentation for each notebook
- Addition of BERT architecture (and potentially beyond)

## References

I'd like to thank the following resources that were influential in my understanding of these networks/papers and consequently the creation of this repository:

- [cs231n (Assignment #3 in particular)](https://cs231n.github.io)
- [PyTorch Machine Translation Article](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Andrej Karpathy: GPT From Scratch](https://youtu.be/kCc8FmEb1nY?si=KekNE5OpIgFVVJsM)
- [Peter Bloem Transformers Article](https://peterbloem.nl/blog/transformers)
- [d2l.ai (11. Attention Mechanisms and Transformers)](https://d2l.ai/chapter_recurrent-modern/index.html)
- [pytorch-seq2seq GitHub Repository](https://github.com/bentrevett/pytorch-seq2seq/tree/main)