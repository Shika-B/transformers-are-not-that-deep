import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from encoder import Encoder
from decoder import Decoder
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
    def __init__(self, tokenizer, file_path, max_len):
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
        lines_eng_tokens_bis = []
        lines_fra_tokens_bis = []

        for line_eng, line_fra in zip(lines_eng_tokens, lines_fra_tokens):
            if max(len(line_eng), len(line_fra)) > max_len:
                continue
            lines_eng_tokens_bis.append(line_eng)
            lines_fra_tokens_bis.append(line_fra)

        self.lines_eng_tokens = lines_eng_tokens_bis
        self.lines_fra_tokens = lines_fra_tokens_bis
        self.len = len(self.lines_fra_tokens)

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


class InverseSqrtLR:
    def __init__(self, optimizer, warmup_steps, peak_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        
        if self.step_num <= self.warmup_steps:
            lr = self.peak_lr * self.step_num / self.warmup_steps
        else:
            lr = self.peak_lr * (self.warmup_steps ** 0.5) / (self.step_num ** 0.5)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


def train(reload_path: str | None = None, save_path: str | None = None):
    # For a reproducible splitting of the dataset in case of reloading
    torch.manual_seed(23)

    num_layers = 4
    num_heads = 8
    d_model = 256
    # Longer sequences (after tokenization) get excluded from the dataset
    max_seq_len = 64
    vocab_size = 10_000
    pad_id = 0

    batch_size = 32
    file_path = "data/eng-fra.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TranslateModel(
        num_layers, num_heads, d_model, max_seq_len, vocab_size, pad_id
    ).to(device)

    if reload_path is not None:
        model.load_state_dict(torch.load(reload_path))

    # For faster/better training
    torch.compile(model, mode="reduce-overhead")
    torch.set_float32_matmul_precision("high")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = InverseSqrtLR(optimizer, warmup_steps=4000, peak_lr=3e-4)
    tokenizer = get_bpe_tokenizer("data/eng-fra.txt", vocab_size=vocab_size)

    def pad_sequences(sequences, max_len, pad_value):
        batch_size = len(sequences)
        padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)

        for i, seq in enumerate(sequences):
            end = min(len(seq), max_len)
            padded[i, :end] = seq[:end]

        return padded

    def collate_fn(batch):
        src_batch, tgt_in_batch, tgt_out_batch = zip(*batch)

        src_batch = pad_sequences(src_batch, max_len=max_seq_len, pad_value=pad_id)
        tgt_in_batch = pad_sequences(
            tgt_in_batch, max_len=max_seq_len, pad_value=pad_id
        )
        tgt_out_batch = pad_sequences(
            tgt_out_batch, max_len=max_seq_len, pad_value=pad_id
        )

        return src_batch, tgt_in_batch, tgt_out_batch

    dataset = FraEngDataset(tokenizer, file_path, max_seq_len)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    print("Number of sentence pairs we are training on:", len(dataset))
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    _test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    print("Loaded data")
    num_epochs = 20

    test_translate = "Longtemps je me suis couché de bonne heure"
    model.train()
    print("Starting training")
    for epoch in range(num_epochs):
        for idx, (src, tgt_in, tgt_out) in enumerate(train_loader):
            model.train()
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
            scheduler.step()

            if epoch + idx != 0 and idx % 1000 == 0:
                print(f"Epoch {epoch}, iteration {idx}")
                translation = translate(
                    test_translate, model, tokenizer, max_len=max_seq_len, device=device
                )
                print(f"Translation of '{test_translate}': {translation}")
                print(f"Last loss: {loss.item()}")

        print("Testing on validation dataset")
        val_loss = 0.0

        model.eval()
        with torch.no_grad():
            for src, tgt_in, tgt_out in val_loader:
                src = src.to(device)
                tgt_in = tgt_in.to(device)
                tgt_out = tgt_out.to(device)

                logits = model(src, tgt_in)

                logits_flat = logits.reshape(-1, vocab_size)
                tgt_flat = tgt_out.reshape(-1)

                loss = criterion(logits_flat, tgt_flat)
                val_loss += loss.item()

        mean_val_loss = val_loss / len(val_loader)
        print("Validation loss:", mean_val_loss)
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            print("Saved model")


def interact(reload_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_layers = 4
    num_heads = 8
    d_model = 256
    # Longer sequences (after tokenization) get excluded from the dataset
    max_seq_len = 64
    vocab_size = 10_000
    pad_id = 0

    model = TranslateModel(
        num_layers, num_heads, d_model, max_seq_len, vocab_size, pad_id
    ).to(device)

    model_state_dict = torch.load(reload_path)
    model.load_state_dict(model_state_dict)
    model.eval()
    tokenizer = get_bpe_tokenizer("data/eng-fra.txt", vocab_size=vocab_size)
    while True:
        to_translate = input("A traduire: ").strip().lower()
        if to_translate in {"exit", "quit"}:
            break
        translated = translate(to_translate, model, tokenizer, device=device)
        print("Traduction: ", translated)


if __name__ == "__main__":
    train(save_path="weights/model.pth")
    # interact("weights/model.pth")
