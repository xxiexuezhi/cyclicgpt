from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
checkpoint = "gpt2litmodel-logs/epoch-34/" #"benchmark-10m/model/" 
config = GPT2Config.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)
tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)

import torch

# 10K subset of PubChem SMILES dataset.
filename = "../data/pubchem-10m.txt"
# Directory to serialize a tokenizer and model.
checkpoint = "../checkpoints/benchmark-10m"
tokenizer_filename = f"cylic_smile_tokenizer_on_pretrain_with_toekn_perm.json" #"../{checkpoint}/tokenizer/tokenizer.json"

# Tokenizer, model, optimizer, scheduler, and trainer hyperparameters.
hyperparams = {"batch_size": 256, "max_epochs": 30, "min_epochs": 15,
               "max_length": 512, "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 6, "n_head": 12, "n_embd": 12 * 48}

gpus = 0  # Specify either a list of GPU devices or an integer (0 for no GPU).
num_workers = 32  # Number of dataloader worker processes.
is_tokenizer_pretrained = True

SUFFIX, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "", "<pad>", "<s>", "</s>", "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID = range(4)


import tqdm

from rdkit.Chem import MolFromSmiles
from rdkit.RDLogger import DisableLog
from rdkit.Chem.Draw import MolsToGridImage
DisableLog("rdApp.*")


model.eval()  # Set the base model to evaluation mode.

generated_smiles_list = []
n_generated = 1000

for _ in tqdm.tqdm(range(n_generated)):
    # Generate from "<s>" so that the next token is arbitrary.
    smiles_start = torch.LongTensor([[1]])
    # Get generated token IDs.
    generated_ids = model.generate(smiles_start,
                                   max_length=hyperparams["max_length"],
                                   do_sample=True, top_p=hyperparams["top_p"],
                                   pad_token_id=tokenizer.eos_token_id)
    # Decode the IDs into tokens and remove "<s>" and "</s>".
    generated_smiles = tokenizer.decode(generated_ids[0],
                                        skip_special_tokens=True)
    generated_smiles_list.append(generated_smiles)




file_path =  f"epoch40_samples.txt"
with open(file_path, "w") as f:
    for s in generated_smiles_list:
        f.write(s + "\n")
    #f.write(str(valid_pct)+"\n")

#valid_molecules = []
#for smiles in generated_smiles_list:
#    molecule = MolFromSmiles(smiles)
#    if molecule is not None:
#        valid_molecules.append(molecule)



#generated_smiles_list[:5]
