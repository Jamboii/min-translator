import string
from typing import Sequence
from collections import Counter

import numpy as np


MAX_LENGTH = 12


def normalize_text(text):
    """
    Normalize the text by converting it to lowercase and removing punctuation.
    :param text: The text to normalize
    :return: The normalized text
    """
    return text.lower().translate(str.maketrans("", "", string.punctuation))


class Tokenizer:
    """Very simple tokenizer for processing basic word-level sequences"""

    def __init__(
        self,
        max_length: int = MAX_LENGTH,
        unknown_token: str = "<UNK>",
        start_token: str = "<SOS>",
        end_token: str = "<EOS>",
    ) -> None:
        """
        Initialize a simple (and not very optimized) word-level tokenizer
        complete with start, end, unknown, and padding tokens.
        :param max_length: The maximum length of a sequence to tokenize
        :param unknown_token: The token to use for words not in the vocabulary
        :param start_token: The token to use for the start of a sequence
        :param end_token: The token to use for the end of a sequence
        """
        self.max_length = max_length
        self.unk_token = unknown_token  # for words out of tokenizer's vocab
        self.sos_token = start_token  # for defining the start of a sentence
        self.eos_token = end_token  # for defining the end of a sentence
        self.pad_token = "<PAD>"  # padding token for padding shorter sequences

        # Map words to indices and vice versa
        self.wtoi = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.sos_token: 2,
            self.eos_token: 3,
        }
        self.itow = {i: w for w, i in self.wtoi.items()}
        self._update_vocab_size()

    def __call__(self, sequence: str) -> Sequence[str]:
        """
        Call method will tokenize a single sequence.
        Assumes fit() was already called.
        :param sequence: The sequence to tokenize
        :return: The tokenized sequence
        """
        return self.tokenize(sequence)

    def _update_vocab_size(self) -> None:
        self.vocab_size = len(self.wtoi)

    def tokenize(self, sequence: str) -> Sequence[str]:
        """
        Tokenize a single sequence. Assumes fit() was already called
        :param sequence: The sequence to tokenize
        :return: The tokenized sequence
        """
        tokens = [self.wtoi[self.pad_token]] * self.max_length
        # Beginning token is a start of a sentence
        tokens[0] = self.wtoi[self.sos_token]
        # Assume the sequence is just word-separated
        ix = 1
        for w in sequence.split(" "):
            tokens[ix] = self.wtoi.get(
                w, self.wtoi[self.unk_token]
            )  # Use unknown token if doesnt exist in our vocab
            ix += 1
        # Index after sentence end is an end of sentence token
        tokens[ix] = self.wtoi[self.eos_token]
        # Returned sequence will have post-padding
        return tokens

    def untokenize(
        self, tokens: Sequence[str], remove_padding_tokens: bool = True
    ) -> str:
        """
        Take a sequence of tokens and use the tokenizer to convert the sequence
        back to a sentence.
        :param tokens: The sequence of tokens to convert
        :param remove_padding_tokens: Whether to remove padding tokens from
        the output
        :return: The untokenized sequence
        """
        words = []
        for token in tokens:
            if remove_padding_tokens and token == self.wtoi[self.pad_token]:
                continue
            word = self.itow[token]
            words.append(word)
        return " ".join(words)

    def fit(self, sequences: Sequence[str], max_vocab_size: int = 500) -> None:
        """
        Create a vocabulary for a list of sequences by word frequency and
        create a mapping to token indices for all words in that vocabulary
        :param sequences: The list of sequences to fit the tokenizer to
        :param max_vocab_size: The maximum size of the vocabulary to create
        :return: None
        """
        # Get the frequencies of the words in all sequences
        word_freqs = Counter([w for s in sequences for w in s.split(" ")])
        # Get the max_vocab_size most frequent words to put into our vocabulary
        vocab_words = sorted(
            list(word_freqs.keys()), key=lambda x: word_freqs[x], reverse=True
        )[:max_vocab_size]
        # Map each word to some integer index
        ix_start = max(self.itow) + 1
        self.wtoi |= {
            w: i
            for w, i in zip(
                vocab_words,
                range(ix_start, ix_start + len(vocab_words)),
            )
        }
        self.itow = {i: w for w, i in self.wtoi.items()}
        self._update_vocab_size()

    def tok_seqs(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Tokenize a list of sequence strings
        :param sequences: The list of sequences to tokenize
        :return: The tokenized sequences
        """
        return np.array([self.tokenize(seq) for seq in sequences])

    def untok_seqs(
        self,
        tokens_list: Sequence[Sequence[str]],
        remove_padding_tokens: bool = True,
    ):
        """
        Untokenize a list of token sequences
        :param tokens_list: The list of token sequences to untokenize
        :param remove_padding_tokens: Whether to remove padding tokens from
        the output
        :return: The untokenized sequences
        """
        sents = []
        for tokens in tokens_list:
            tokens = list(tokens)
            sent = self.untokenize(
                tokens,
                remove_padding_tokens=remove_padding_tokens,
            )
            sents.append(sent)
        return sents
