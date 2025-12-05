from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFD, StripAccents, Lowercase, NFC
from tokenizers.pre_tokenizers import ByteLevel, Whitespace

from pathlib import Path


def get_bpe_tokenizer(
    text_file: str, vocab_size: int = 10000, save_path: str = "tokenizer.json"
):
    path = Path(save_path)
    if path.is_file():
        tokenizer = Tokenizer.from_file(save_path)
        return tokenizer

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.normalizer = Sequence([NFD(), StripAccents(), Lowercase(), NFC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )

    tokenizer.train([text_file], trainer)
    tokenizer.save(save_path)

    return tokenizer
