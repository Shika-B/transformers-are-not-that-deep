import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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
    translation = re.sub(" ", "", translation)
    translation = re.sub('Ġ', ' ', translation)
    # translation = re.sub("\\.| |,", "", translation)
    # translation = re.sub("Ġ|'", ' ', translation)

    return translation.strip()


def train(reload_path: str | None = None, save_path: str | None = None):
    num_layers = 4
    num_heads = 4
    d_model = 128
    # Longer sequences (after tokenization) get excluded from the dataset
    max_seq_len = 64
    vocab_size = 8_000
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

    base_lr = 5e-4
    warmup_steps = 25
    total_steps = 500  # depends on dataset size / batch size / epochs
    weight_decay = 0.01

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
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
    print("Number of sentence pairs we are training on:", len(dataset))

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    print("Loaded data")
    num_epochs = 20

    test_translate = "c est difficile de te comprendre"
    model.train()
    print("Starting training")
    for epoch in range(0, num_epochs):
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

        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            print("Saved model")


def interact(reload_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_layers = 4
    num_heads = 4
    d_model = 128
    max_seq_len = 64
    vocab_size = 8_000
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
        print("Traduction:", translated)

def gen_test_answers(reload_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_layers = 4
    num_heads = 4
    d_model = 128
    max_seq_len = 64
    vocab_size = 8_000
    pad_id = 0

    model = TranslateModel(
        num_layers, num_heads, d_model, max_seq_len, vocab_size, pad_id
    ).to(device)

    model_state_dict = torch.load(reload_path)
    model.load_state_dict(model_state_dict)
    model.eval()
    tokenizer = get_bpe_tokenizer("data/eng-fra.txt", vocab_size=vocab_size)

    result = "ID,TARGET\n"
    with open('data/fra.csv') as file:
        lines = file.read().split('\n')[1:]
        for line in lines:
            idx, content = line.split(',')
            translated = translate(content, model, tokenizer, device=device)
            result += f"{idx}, {translated}\n"
            print(idx)
    
    with open('data/results.csv', 'w+') as file:
        file.write(result)
 
if __name__ == "__main__":
    # train(save_path="weights/model128-4-4-8000.pth", reload_path="weights/model128-4-4-8000.pth")
    interact("weights/model128-4-4-8000.pth")
    g# en_test_answers(reload_path="weights/model128-4-4-8000.pth")