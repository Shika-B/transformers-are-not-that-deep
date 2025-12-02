import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import TransformerBlock, SinusoidalPositionalEncoding
from encoder import Encoder
from decoder import Decoder
from my_utils import _init_weights
from bpe import get_bpe_tokenizer

import re


class TranslateModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        max_seq_len: int,
        vocab_size: int,
        pad_id: int,
        embed_drop: float = 0.1,
        atn_drop: float = 0.1,
        out_drop: float = 0.1,
        mlp_drop: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        self.encoder = Encoder(
            num_layers,
            num_heads,
            d_model,
            max_seq_len,
            vocab_size,
            embed_drop,
            atn_drop,
            out_drop,
            mlp_drop,
            bias,
        )

        self.decoder = Decoder(
            num_layers,
            num_heads,
            d_model,
            max_seq_len,
            vocab_size,
            embed_drop,
            atn_drop,
            out_drop,
            mlp_drop,
            bias,
        )

        self.pad_id = pad_id

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Input:
        - src has shape (batch_size, src_len)
        - tgt has shape (batch_size, tgt_len)

        Output:
        Returns logits with shape (batch_size, tgt_len, vocab_size)
        """
        # [batch_size, 1, 1, src_len]
        src_pad_mask = (src == self.pad_id)[:, None, None, :]
        # [batch_size, 1, 1, tgt_len]
        tgt_pad_mask = (tgt == self.pad_id)[:, None, None, :]

        encoded = self.encoder(src, src_pad_mask)
        decoded = self.decoder(tgt, encoded, tgt_pad_mask, src_pad_mask)

        return decoded


class FraEngDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path):
        super().__init__()
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().strip().split("\n")
            lines_eng = []
            lines_fra = []
            for line in lines:
                cols = line.split("\t")
                lines_eng.append(cols[0])
                lines_fra.append(cols[1])

        lines_eng_tokens = [
            tokenizer.encode(
                "<bos> " + line + " <eos>",
            ).ids
            for line in lines_eng
        ]
        lines_fra_tokens = [
            tokenizer.encode(
                "<bos> " + line + " <eos>",
            ).ids
            for line in lines_fra
        ]

        self.lines_eng_tokens = lines_eng_tokens
        self.lines_fra_tokens = lines_fra_tokens
        self.len = len(lines_fra_tokens)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        src = self.lines_fra_tokens[idx]
        tgt_in = self.lines_eng_tokens[idx][:-1]
        tgt_out = self.lines_eng_tokens[idx][1:]

        return torch.tensor(src), torch.tensor(tgt_in), torch.tensor(tgt_out)


def translate(sentence, model, tokenizer, max_len=40, device="cpu"):
    model.eval()

    src_tokens = tokenizer.encode("<bos> " + sentence + " <eos>").ids
    src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)

    src_mask = (src_tensor == tokenizer.token_to_id("<pad>")).unsqueeze(1).unsqueeze(2)

    encoder_outputs = model.encoder(src_tensor, src_mask)

    tgt_tokens = [tokenizer.token_to_id("<bos>")]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long, device=device)

        tgt_mask = torch.zeros_like(tgt_tensor, dtype=torch.bool, device=device)
        # [1, seq_len, vocab_size]
        logits = model.decoder(tgt_tensor, encoder_outputs, tgt_mask, src_mask)

        next_token_id = logits[0, -1].argmax().item()
        tgt_tokens.append(next_token_id)

        if next_token_id == tokenizer.token_to_id("<eos>"):
            break

    translation = tokenizer.decode(tgt_tokens, skip_special_tokens=True)

    # Fix apostrophes and punctuation extra spaces
    translation = re.sub(r"\s+'\s+", "'", translation)
    translation = re.sub(r"\s+([?.!,;:])", r"\1", translation)

    return translation


def train(reload_path: str | None = None, save_path: str | None = None):
    num_layers = 4
    num_heads = 4
    d_model = 256
    max_seq_len = 128
    vocab_size = 10_000
    pad_id = 0
    file_path = "data/eng-fra.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TranslateModel(
        num_layers, num_heads, d_model, max_seq_len, vocab_size, pad_id
    ).to(device)

    if reload_path is not None:
        model.load_state_dict(torch.load(reload_path))

    model = torch.compile(model, mode="default")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    tokenizer = get_bpe_tokenizer("data/eng-fra.txt")

    def collate_fn(batch):
        src_batch, tgt_in_batch, tgt_out_batch = zip(*batch)

        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
        tgt_in_batch = pad_sequence(tgt_in_batch, batch_first=True, padding_value=0)
        tgt_out_batch = pad_sequence(tgt_out_batch, batch_first=True, padding_value=0)

        return src_batch, tgt_in_batch, tgt_out_batch

    train_loader = DataLoader(
        FraEngDataset(tokenizer, file_path),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    num_epochs = 10

    test_translate = "Longtemps je me suis couché de bonne heure"
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0

        for idx, (src, tgt_in, tgt_out) in enumerate(train_loader):
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            optimizer.zero_grad()

            logits = model(src, tgt_in)

            # reshape to (batch_size * tgt_len, vocab_size)
            logits_flat = logits.reshape(-1, vocab_size)
            # reshape targets to (batch_size * tgt_len)
            tgt_flat = tgt_out.reshape(-1)

            loss = criterion(logits_flat, tgt_flat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if idx != 0 and idx % 1000 == 0:
                print(f"Epoch {epoch}, iteration {idx}")
                translation = translate(
                    test_translate, model, tokenizer, max_len=128, device=device
                )
                print(f"Translation of '{test_translate}: {translation}")
                print(f"Last loss: {loss.item()}")
        avg_loss = total_loss / len(train_loader)
        print(f"Average loss {avg_loss}")

        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            print("Saved model")


if __name__ == "__main__":
    train(save_path="weights/model.pth")
