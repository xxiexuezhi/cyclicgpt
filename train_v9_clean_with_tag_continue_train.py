from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
import pytorch_lightning as pl
import torch
import random


# 10K subset of PubChem SMILES dataset.
#filename = "../data/pubchem-10k.txt"
#filename = "../data/cyclic_smiles_only.csv"




## loaded last checkpoint
checkpoint = "../checkpoints/benchmark-5m"

config = GPT2Config.from_pretrained(checkpoint)


### post train. need to read from previous epoch
checkpoint2 = "gpt2litmodel-logs/epoch-170/"#"../notebook_v_pretrain_with_pretrain_tokenizer/gpt2litmodel-logs/epoch_1/"
model = GPT2LMHeadModel.from_pretrained(checkpoint2)
#model = GPT2LMHeadModel(config)
#tokenizer = PreTrainedTokenizerFast.from_pretrained("cyclic_peptide_bpe_tokenizer.json")

"""SMILES-based tokenization utilities.
"""

__all__ = ("PAD_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "UNK_TOKEN", "SUFFIX",
           "SPECIAL_TOKENS", "PAD_TOKEN_ID", "BOS_TOKEN_ID", "EOS_TOKEN_ID",
           "UNK_TOKEN_ID", "SMILESBPETokenizer", "SMILESAlphabet")

from collections.abc import Collection, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from tokenizers import AddedToken, Tokenizer
from tokenizers import decoders, models, normalizers, processors, trainers
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast


SUFFIX, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "", "<pad>", "<s>", "</s>", "<unk>"
#SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<perm=low>", "<perm=high>", "<perm=unknown>"]
PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID = range(4)


class SMILESBPETokenizer(BaseTokenizer):
    """Tokenizes SMILES strings and applies BPE.

    Args:
        vocab (`str` or `dict`, optional, defaults to `None`):
            Token vocabulary.
        merges (`str` or `dict` or `tuple`, optional, defaults to `None`):
            BPE merges.
        unk_token (`str` or `tokenizers.AddedToken`, optional, defaults to "<unk>")
        suffix (`str`, defaults to "")
        dropout (`float`, defaults to `None`)

    Examples:
        >>> tokenizer = SMILESBPETokenizer()
        >>> tokenizer.train("path-to-smiles-strings-file")
        Tokenization logs...
        >>> tokenizer.save_model("checkpoints-path")
        >>> same_tokenizer = SMILESBPETokenizer.from_file("checkpoints-path/vocab.json",
        ...                                               "checkpoints-path/merges.txt")
    """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        merges: Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]] = None,
        unk_token: Union[str, AddedToken] = "<unk>",
        suffix: str = SUFFIX,
        dropout: Optional[float] = None,
    ) -> None:
        unk_token_str = str(unk_token)

        tokenizer = Tokenizer(models.BPE(vocab, merges, dropout=dropout,
                                         unk_token=unk_token_str,
                                         end_of_word_suffix=suffix))

        if tokenizer.token_to_id(unk_token_str) is not None:
            tokenizer.add_special_tokens([unk_token_str])

        tokenizer.normalizer = normalizers.Strip(left=False, right=True)
        tokenizer.decoder = decoders.Metaspace(add_prefix_space=True)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[(BOS_TOKEN, BOS_TOKEN_ID), (EOS_TOKEN, EOS_TOKEN_ID)])

        parameters = {"model": "BPE", "unk_token": unk_token, "suffix": suffix,
                      "dropout": dropout}

        super().__init__(tokenizer, parameters)

    @classmethod
    def from_file(cls, vocab_filename: str, merges_filename: str, **kwargs) \
            -> "SMILESBPETokenizer":
        vocab, merges = models.BPE.read_file(vocab_filename, merges_filename)
        return cls(vocab, merges, **kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 600,#1_000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = None,
        limit_alphabet: int = 200,
        initial_alphabet: List[str] = None,
        suffix: Optional[str] = SUFFIX,
        show_progress: bool = True,
    ) -> None:
        special_tokens = special_tokens or SPECIAL_TOKENS
        initial_alphabet = initial_alphabet or []

        trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                      min_frequency=min_frequency,
                                      special_tokens=special_tokens,
                                      limit_alphabet=limit_alphabet,
                                      initial_alphabet=initial_alphabet,
                                      end_of_word_suffix=suffix,
                                      show_progress=show_progress)
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Iterator,
        vocab_size: int = 1_000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = None,
        limit_alphabet: int = 200,
        initial_alphabet: List[str] = None,
        suffix: Optional[str] = SUFFIX,
        show_progress: bool = True,
    ) -> None:
        special_tokens = special_tokens or SPECIAL_TOKENS
        initial_alphabet = initial_alphabet or []

        trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                      min_frequency=min_frequency,
                                      special_tokens=special_tokens,
                                      limit_alphabet=limit_alphabet,
                                      initial_alphabet=initial_alphabet,
                                      end_of_word_suffix=suffix,
                                      show_progress=show_progress)
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

    @staticmethod
    def get_hf_tokenizer(
        tokenizer_file: str,
        special_tokens: Optional[Dict[str, str]] = None,
        model_max_length: int = 256, #512,
        *init_inputs, **kwargs
    ) -> PreTrainedTokenizerFast:
        """Gets HuggingFace tokenizer from the pretrained `tokenizer_file`. Optionally,
        appends `special_tokens` to vocabulary and sets `model_max_length`.
        """
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file,
                                            *init_inputs, **kwargs)
        special_tokens = special_tokens or dict(zip(
            ["pad_token", "bos_token", "eos_token", "unk_token"],
            SPECIAL_TOKENS))
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.model_max_length = model_max_length
        return tokenizer


@dataclass(init=True, eq=False, repr=True, frozen=True)
class SMILESAlphabet(Collection):
    atoms: FrozenSet[str] = frozenset([
        'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bh',
        'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr',
        'Cs', 'Cu', 'Db', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fm', 'Fr', 'Ga',
        'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K',
        'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na',
        'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd',
        'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rh', 'Rn',
        'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb',
        'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Xe', 'Y', 'Yb',
        'Zn', 'Zr'
    ])

    # Bonds, charges, etc.
    non_atoms: FrozenSet[str] = frozenset([
        '-', '=', '#', ':', '(', ')', '.', '[', ']', '+', '-', '\\', '/', '*',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        '@', 'AL', 'TH', 'SP', 'TB', 'OH',
    ])

    additional: FrozenSet[str] = frozenset()

    def __contains__(self, item: Any) -> bool:
        return item in self.atoms or item in self.non_atoms

    def __iter__(self):
        return (token for token in chain(self.atoms, self.non_atoms))

    def __len__(self) -> int:
        return len(self.atoms) + len(self.non_atoms) + len(self.additional)

    def get_alphabet(self) -> Set[str]:
        alphabet = set()
        for token in self.atoms:
            if len(token) > 1:
                alphabet.update(list(token))
                alphabet.add(token[0].lower())
            else:
                alphabet.add(token)
                alphabet.add(token.lower())
        for token in chain(self.non_atoms, self.additional):
            if len(token) > 1:
                alphabet.update(list(token))
            else:
                alphabet.add(token)
        return alphabet



from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import ShuffleSplit
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader



from typing import Union, List
from torch.utils.data import Dataset
from pathlib import Path
from functools import partial
import torch
from transformers import PreTrainedTokenizerFast

class LMDataset(Dataset):
    """Accepts a file path OR list of SMILES strings for language modeling."""

    def __init__(self, smiles: Union[str, List[str]], tokenizer):
        if isinstance(smiles, str):  # it's a file path
            self.smiles_strings = Path(smiles).read_text(encoding='ascii').splitlines()
        else:  # it's a list of SMILES
            self.smiles_strings = smiles

        self.tokenizer = tokenizer

        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            self._encode = partial(self.tokenizer.__call__, truncation=True)
            self._id_key = "input_ids"
        else:
            self._encode = self.tokenizer.encode
            self._id_key = "ids"

    def __len__(self):
        return len(self.smiles_strings)

    def __getitem__(self, i):
        encodings = self._encode(self.smiles_strings[i])
        return torch.LongTensor(getattr(encodings, self._id_key))

@dataclass(init=True, repr=True, eq=False, frozen=False)
class LMDataModule(LightningDataModule):
    """Lightning data module for autoregressive language modeling with train/val/test split."""

    filename: str
    tokenizer: BaseTokenizer
    batch_size: int = 128
    num_workers: int = 0
    collate_fn: Union[None, Literal["default"], Callable] = "default"
    val_size: float = 0.08
    test_size: float = 0.02
    seed: int = 42
    #self.seed = 42

    def __post_init__(self) -> None:
        super().__init__()
        #self.seed = 42
        if self.collate_fn == "default":
            self.collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def setup(self, stage: Optional[str] = None) -> None:
        all_smiles = Path(self.filename).read_text(encoding='ascii').splitlines()
        random.seed(self.seed)
        random.shuffle(all_smiles)

        n_total = len(all_smiles)
        n_test = int(self.test_size * n_total)
        n_val = int(self.val_size * n_total)

        test_smiles = all_smiles[:n_test]
        val_smiles = all_smiles[n_test:n_test + n_val]
        train_smiles = all_smiles[n_test + n_val:]

        self.train_dataset = LMDataset(train_smiles, self.tokenizer)
        self.val_dataset = LMDataset(val_smiles, self.tokenizer)
        self.test_dataset = LMDataset(test_smiles, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)


class GPT2LitModel(pl.LightningModule):
    def __init__(self, transformer, batch_size: int, learning_rate: float,
                 final_learning_rate: float, weight_decay: float, adam_eps: float,
                 adam_betas: tuple, scheduler_T_max: int,
                 save_model_every: int = 10_000, checkpoint: str = ""):
        super().__init__()
        self.save_hyperparameters(ignore=("transformer", "save_model_every", "checkpoints"))
        self.transformer = transformer
        self.save_model_every = save_model_every
        self.checkpoint = checkpoint or "./gpt2litmodel-logs"
        self._epoch_val_losses = []
        self._epoch_train_losses = []
        self._epoch_ppl_steps = []
        #self.start_epoch=201
        self._start_epoch = 170
        # Store intermediate outputs here
        self._epoch_ppl_steps = []

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


    def on_train_start(self):
        # 🧠 This is the *reliable* way to resume from a custom epoch
        self.trainer.fit_loop.epoch_progress.current.completed = self._start_epoch

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        ppl = torch.exp(loss)

        self.log("ppl", ppl, on_step=True, on_epoch=False, prog_bar=True)

        # Save for epoch-end logging
        self._epoch_ppl_steps.append(ppl.detach())

        if self.save_model_every > 0 and batch_idx % self.save_model_every == 0:
            self.transformer.save_pretrained(self.checkpoint)

        return loss  # just return the loss now

    def on_train_epoch_end(self):
        if self.save_model_every > 0:
            #self.transformer.save_pretrained(self.checkpoint)
            save_path = f"{self.checkpoint}/epoch-{self.current_epoch + 1}"
            self.transformer.save_pretrained(save_path)


        if self._epoch_ppl_steps:
            mean_ppl = torch.stack(self._epoch_ppl_steps).mean()
            self.log("ppl_epoch", mean_ppl, on_step=False, on_epoch=True, prog_bar=True)
            self._epoch_ppl_steps.clear()


    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        self._epoch_val_losses.append(loss.detach())
        self.log("val_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self._epoch_val_losses:
            avg_val_loss = torch.stack(self._epoch_val_losses).mean()
            self.log("val_loss_epoch", avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self._epoch_val_losses.clear()

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        self.log("test_loss_step", loss, on_step=True, on_epoch=False)
        return loss

    def on_test_epoch_end(self):
        # This will automatically aggregate test_loss_step over all steps
        pass  #


    def configure_optimizers(self):
        parameters = self.named_parameters()
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]
        optimizer = torch.optim.Adam(
            grouped_parameters, lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_eps, betas=self.hparams.adam_betas
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.hparams.scheduler_T_max,
            eta_min=self.hparams.final_learning_rate
        )

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler,
                                 'interval': 'step', 'frequency': 1}}













# 10K subset of PubChem SMILES dataset.
#filename = "../data/pubchem-10k.txt"
#filename = "../data/cyclic_smiles_only.csv"
filename = "cyclic_smiles_with_perm_non_augmented.txt" #"cyclic_smiles_with_perm_augmented.txt" #"filtered_smiles_len30.csv"  #"cyclic_smiles_only_augmented.csv"
# Directory to serialize a tokenizer and model.
checkpoint = "benchmark-10m"
tokenizer_filename = "../notebook_v_pretrain_with_pretrain_tokenizer/gpt2litmodel-logs/epoch_1/tokenizer.json" #"cylic_smile_tokenizer_on_cyclic_aug_toy.json" #"cylic_smile_tokenizer.json" #"../checkpoints/{checkpoint}/tokenizer.json"

# Tokenizer, model, optimizer, scheduler, and trainer hyperparameters.
hyperparams = {"batch_size": 32, "max_epochs": 300, "min_epochs": 6,
               "max_length": 256, "learning_rate": 2e-6, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 4, "n_head": 12, "n_embd": 12 * 48}

gpus = 0  # Specify either a list of GPU devices or an integer (0 for no GPU).
num_workers = 32  # Number of dataloader worker processes.
is_tokenizer_pretrained = True






print(tokenizer_filename)


#tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
#    tokenizer_filename, model_max_length=hyperparams["max_length"])

tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
    tokenizer_filename,
    model_max_length=hyperparams["max_length"],
    special_tokens={
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "additional_special_tokens": ["<perm=low>", "<perm=high>", "<perm=unknown>"]
    }
)




datamodule = LMDataModule(filename, tokenizer,
                              batch_size=hyperparams["batch_size"],
                              num_workers=num_workers)
# datamodule.setup()

# batch = next(iter(datamodule.train_dataloader()))
# pprint(batch)




# === Freeze early layers ===
#for name, param in model.named_parameters():
#    if any(name.startswith(layer) for layer in [
#        "transformer.wte", "transformer.h.0", "transformer.h.1", "transformer.h.2"
#    ]):
#        param.requires_grad = False


# Freeze embeddings and first 3 transformer blocks (out of 4 total)
for name, param in model.named_parameters():
    if any(name.startswith(layer) for layer in [
        "transformer.wte", 
        "transformer.wpe",  # positional embeddings (optional)
        "transformer.h.0", 
        "transformer.h.1", 
        "transformer.h.2"
    ]):
        param.requires_grad = False


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping






from pytorch_lightning.callbacks import Callback
from rdkit import Chem
import torch
import os

class ValiditySamplingCallback(Callback):
    def __init__(self, tokenizer, n_samples=20, output_dir="generated_samples", max_length=256, top_p=0.95):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.output_dir = output_dir
        self.max_length = max_length
        self.top_p = top_p
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        model = pl_module.transformer
        model.eval()
        valid_count = 0
        total = self.n_samples
        smiles_list = []

        for _ in range(self.n_samples):
            start_token = torch.tensor([[self.tokenizer.bos_token_id]]).to(model.device)
            generated_ids = model.generate(
                start_token,
                max_length=self.max_length,
                do_sample=True,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            smiles_list.append(decoded)
            if Chem.MolFromSmiles(decoded):
                valid_count += 1

        epoch = trainer.current_epoch
        valid_pct = valid_count / total * 100
        print(f"[Callback] Epoch {epoch}: {valid_count}/{total} valid SMILES ({valid_pct:.2f}%)")

        file_path = os.path.join(self.output_dir, f"epoch_{epoch:02d}_samples.txt")
        with open(file_path, "w") as f:
            for s in smiles_list:
                f.write(s + "\n")
            #f.write(str(valid_pct)+"\n")
        print(f"[Callback] Saved {total} SMILES to {file_path}")



from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("lightning_logs", name="cyclic_gpt")





checkpoint_cb = ModelCheckpoint(f"{checkpoint}/model/")
early_stopping_ppl = EarlyStopping(
    monitor="ppl_epoch",
    patience=10,
    min_delta=1e-4, #5e-3,
    check_finite=True,
    stopping_threshold=1.1,
    divergence_threshold=hyperparams["vocab_size"] / 10,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=True,
)

early_stopping_val = EarlyStopping(
    monitor="val_loss_epoch",     # ← this is what you log in on_validation_epoch_end
    patience=5,
    min_delta=1e-4,
    check_finite=True,
    mode="min",
    verbose=True,
    check_on_train_epoch_end=True,
)

trainer = Trainer(
    #gpus=gpus,
    logger=logger,
    strategy="auto",#"dp",
    callbacks=[checkpoint_cb, early_stopping_val, ValiditySamplingCallback(tokenizer, n_samples=1000, output_dir="generated_samples")],
    max_epochs=hyperparams["max_epochs"],
    min_epochs=hyperparams["min_epochs"],
    val_check_interval=0.4,
    limit_train_batches=0.2,
    log_every_n_steps=5,
)
lit_model = GPT2LitModel(
    model,
    batch_size=hyperparams["batch_size"],
    learning_rate=hyperparams["learning_rate"],
    final_learning_rate=hyperparams["final_learning_rate"],
    weight_decay=hyperparams["weight_decay"],
    adam_eps=hyperparams["adam_eps"],
    adam_betas=hyperparams["adam_betas"],
    scheduler_T_max=hyperparams["scheduler_T_max"],
)


trainer.fit_loop.epoch_progress.current.completed = 201  # 👈 manually resume from epoch 201


trainer.fit(lit_model, datamodule)

lit_model.transformer.save_pretrained(f"{checkpoint}/model/")
