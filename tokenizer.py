class TokenizerBPE():
    """
    Implements Byte-Pair encoding* algorithm and various utilities around it.
        *See https://en.wikipedia.org/wiki/Byte-pair_encoding
    """
    def __init__(self, max_tokens: int):
        """ 
        max_tokens serves as a hint for the desired vocabulary size.
        It may not be respected.
        In particular, the vocabulary will always have atleast 256 tokens, 
        as there is 256 possible values for a byte.

        """
        if max_tokens < 256:
            max_tokens = 256
        self.max_tokens = max_tokens
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_ranks = {}

    def train(self, corpus: list[str]):
        
        fresh_token = 256
        while fresh_token < self.max_tokens:
            fresh_token += 1

        processed_text = []
        for text in corpus:
            for i, char in enumerate(text):
                if char == " " and i != 0:
                    processed_text.append("_")
                if char != " ":
                    processed_text.append(char)
            processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including "_" if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted(set(processed_text))
            if char not in unique_chars
        )
        if "_" not in unique_chars:
            unique_chars.append("_")
        
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}
