import os
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling
)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from rdkit import Chem
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import ShuffleSplit
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader




# === Config ===
checkpoint_path = "../checkpoints/benchmark-5m"
csv_file = "cyclic_smiles_only.csv" #"CycPeptMPDB_Peptide_All.csv"
output_sample_dir = "generated_samples"
tokenizer_file = f"{checkpoint_path}/tokenizer.json"
os.makedirs(output_sample_dir, exist_ok=True)

hyperparams = {
    "batch_size": 16,
    "max_epochs": 300,
    "min_epochs": 15,
    "max_length": 128,
    "learning_rate": 1e-5,
    "final_learning_rate": 5e-7,
    "weight_decay": 0.0,
    "adam_eps": 1e-8,
    "adam_betas": (0.9, 0.999),
    "scheduler_T_max": 150_000,
    "top_p": 0.95,
}


# === Load model and tokenizer ===
config = GPT2Config.from_pretrained(checkpoint_path)
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
tokenizer.add_special_tokens({
    "pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"
})
tokenizer.model_max_length = hyperparams["max_length"]


# === Freeze early layers ===
for name, param in model.named_parameters():
    if any(name.startswith(layer) for layer in [
        "transformer.wte", "transformer.h.0", "transformer.h.1", "transformer.h.2"
    ]):
        param.requires_grad = False


# === Dataset ===

#@dataclass(init=True, eq=False, repr=True, frozen=False)
class LMDataset(Dataset):
    """Simple sequential dataset for autoregressive language modeling.
    """

    filename: str
    tokenizer: BaseTokenizer

    def __post_init__(self) -> None:
        self.smiles_strings = Path(self.filename).read_text(encoding='ascii').splitlines()

        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            self._encode = partial(self.tokenizer.__call__, truncation=True)
            self._id_key = "input_ids"
        else:
            self._encode = self.tokenizer.encode
            self._id_key = "ids"

    def __len__(self) -> int:
        return len(self.smiles_strings)

    def __getitem__(self, i: int) -> torch.Tensor:
        encodings = self._encode(self.smiles_strings[i])
        return torch.LongTensor(getattr(encodings, self._id_key))




class LMDataModule(pl.LightningDataModule):
    def __init__(self, filename, tokenizer, batch_size=16, num_workers=4):
        super().__init__()
        self.filename = filename
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = LMDataset(self.filename, self.tokenizer)

    def train_dataloader(self):
        collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=self.num_workers)


# === Model wrapper ===
class GPT2LitModel(pl.LightningModule):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    def training_step(self, batch, batch_idx):
        outputs = self(**{"input_ids": batch})
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=hyperparams["learning_rate"],
                                      eps=hyperparams["adam_eps"],
                                      betas=hyperparams["adam_betas"],
                                      weight_decay=hyperparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hyperparams["scheduler_T_max"],
            eta_min=hyperparams["final_learning_rate"]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# === Sample Callback ===
class SampleSMILESEpochCallback(Callback):
    def __init__(self, tokenizer, output_dir, n_samples=20, max_length=128):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.n_samples = n_samples
        self.max_length = max_length
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        model = pl_module.transformer
        model.eval()
        generated = []

        for _ in range(self.n_samples):
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(model.device)
            output = model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                do_sample=True,
                top_p=hyperparams["top_p"],
                pad_token_id=self.tokenizer.eos_token_id
            )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            mol = Chem.MolFromSmiles(decoded)
            if mol is not None:
                generated.append(decoded)

        save_path = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch:02d}_samples.txt")
        with open(save_path, "w") as f:
            for s in generated:
                f.write(s + "\n")
        print(f"[Callback] Saved {len(generated)} valid SMILES to {save_path}")


# === Train ===
datamodule = LMDataModule(csv_file, tokenizer,
                          batch_size=hyperparams["batch_size"])

lit_model = GPT2LitModel(model)

callbacks = [
    ModelCheckpoint(dirpath=checkpoint_path, save_top_k=1, monitor="train_loss"),
    EarlyStopping(monitor="train_loss", patience=4),
    SampleSMILESEpochCallback(tokenizer, output_sample_dir)
]

trainer = Trainer(
    strategy="auto",
    callbacks=callbacks,
    max_epochs=hyperparams["max_epochs"],
    min_epochs=hyperparams["min_epochs"],
    log_every_n_steps=5,
)

trainer.fit(lit_model, datamodule)


