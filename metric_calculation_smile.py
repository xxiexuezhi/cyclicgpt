import torch
import warnings
import numpy as np
import pandas as pd
from typing import List
#from loguru import logger
from rdkit import Chem, rdBase, DataStructs
import pickle
import gzip
#from utils.sascore.sascorer import calculateScore
from rdkit.Chem import AllChem, Descriptors
#from utils.utils import mapper

#from utils.helm_utils import get_cycpep_smi_from_helm

rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Metrics:
    def __init__(self, train_smiles_file, n_jobs=100):
        # Load and canonicalize training SMILES
        train_smi_raw = pd.read_csv(train_smiles_file, header=None)[0].tolist()
        self.train_set_cano_smi = [canonic_smiles(smi) for smi in train_smi_raw if canonic_smiles(smi) is not None]
        self.n_jobs = n_jobs
        self.input_type = 'smiles'
        self.ref_fps = np.vstack([
            fingerprints_from_mol(Chem.MolFromSmiles(smi)) 
            for smi in self.train_set_cano_smi
        ])

    def get_metrics(self, generated_smiles: List[str]):
        # Convert to molecules
        mols = [Chem.MolFromSmiles(smi) if smi else None for smi in generated_smiles]
        is_valid = [1 if mol else 0 for mol in mols]
        validity = sum(is_valid) / len(generated_smiles) if generated_smiles else 0.0

        valid_canon_smiles = [Chem.MolToSmiles(mol) for mol in mols if mol]
        uniqueness = len(set(valid_canon_smiles)) / len(valid_canon_smiles) if valid_canon_smiles else 0.0

        uniq_smis = list(set(valid_canon_smiles))
        uniq_mols = [Chem.MolFromSmiles(smi) for smi in uniq_smis]

        if uniq_mols:
            fps = np.vstack([fingerprints_from_mol(mol) for mol in uniq_mols])
            diversity = 1 - average_agg_tanimoto(fps, fps, agg='mean', p=1)
            snn = average_agg_tanimoto(self.ref_fps, fps, agg='max', p=1)
        else:
            diversity = 0.0
            snn = 0.0

        # Novelty
        gen_smiles = mapper(self.n_jobs)(canonic_smiles, valid_canon_smiles)
        gen_smiles_set = set([s for s in gen_smiles if s])  # remove None
        train_set = set(self.train_set_cano_smi)
        novelty = len(gen_smiles_set - train_set) / len(gen_smiles_set) if gen_smiles_set else 0.0

        print(f"validity\tuniqueness\tdiversity\tsnn\tnovelty")
        print(f"{validity:.3f}\t{uniqueness:.3f}\t{diversity:.3f}\t{snn:.3f}\t{novelty:.3f}")
        return {
            "validity": validity,
            "uniqueness": uniqueness,
            "diversity": diversity,
            "snn": snn,
            "novelty": novelty,
        }



def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    """ Create ECFP fingerprint of a molecule """
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)


def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg='max', device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_cano_smi_from_helm(helm):
    smi = get_cycpep_smi_from_helm(helm)
    return canonic_smiles(smi)

def get_cano_smi_from_helms(helms, n_jobs=100):
    all_smiles =  mapper(n_jobs)(get_cano_smi_from_helm, helms)
    return all_smiles


class SAScorer:
    def __init__(self, model_path='utils/sascore/SA_score_prediction.pkl.gz', input_type='smiles'):
        self.clf = pickle.load(gzip.open(model_path, "rb"))
        self.input_type = 'helm' if input_type != 'smiles' else 'smiles'

    def __call__(self, helms: List):
        scores = self.get_scores(helms)
        return scores, scores

    @staticmethod
    def _get_descriptors_from_smiles(smiles: List, radius=3, size=4096):  #
        """
            Add fingerprints together with SAscore and molecular weights
        """
        fps = []
        valid_mask = []
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile) if smile is not None else None
            valid_mask.append(int(mol is not None))
            fp = fingerprints_from_mol(mol, radius, size=size) if mol else np.zeros((1, size))
            others = np.array([calculateScore(mol), Descriptors.ExactMolWt(mol)]) if mol else np.zeros(2)
            prop_np = np.concatenate([others.T, fp.T[:, 0]])
            fps.append(prop_np)

        return fps, valid_mask
    
    def get_scores(self, input_seqs: List, valid_only=False):
        smiles = [get_cycpep_smi_from_helm(helm) for helm in input_seqs] if self.input_type == 'helm' else input_seqs
        descriptors, valid_mask = self._get_descriptors_from_smiles(smiles)
        scores = self.clf.predict_proba(descriptors)[:, 1]
        if valid_only:  # filter by valid mask
            return np.float32([scores[i] for i in range(len(scores)) if valid_mask[i]])
        return np.float32(scores * np.array(valid_mask))
import os
import time
from datetime import timedelta
import numpy as np
import torch
import yaml
#from easydict import EasyDict
#from loguru import logger
from multiprocessing import Pool

def time_since(start_time):
    seconds = int(time.time() - start_time)
    return str(timedelta(seconds=seconds))


def get_path(base_dir, base_name, suffix):
    return os.path.join(base_dir, base_name + suffix)


def set_random_seed(seed, device):
    """
    Set the random seed for Numpy and PyTorch operations
    Args:
        seed: seed for the random number generators
        device: "cpu" or "cuda"
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = EasyDict(config)
    return config

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


# Assuming you have:
# - A list of generated SMILES: generated = [...]
# - A CSV file with 1 column of training SMILES: training_data.csv


def load_smiles_from_txt(file_path):
    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list

import sys
#name_input = sys.argv[1]
def evaluate_and_save(folder="generated_samples", output_csv="metrics_results.csv",
                      file_prefix="epoch_", file_suffix="_samples.txt",
                      train_file="cyclic_smiles_only.csv", epochs=range(100, 129)):

    metric_evaluator = Metrics(train_smiles_file=train_file)
    all_results = []

    for epoch in epochs:
        filename = os.path.join(folder, f"{file_prefix}{epoch}{file_suffix}")
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue
        smiles_list = load_smiles_from_txt(filename)
        print(f"Evaluating: {filename}")
        metrics = metric_evaluator.get_metrics(smiles_list)
        print(metrics)
        all_results.append({
            "epoch": epoch,
            **metrics
        })

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to: {output_csv}")


# === Run the script (uncomment to run directly) ===
# if __name__ == "__main__":
#     evaluate_and_save()

evaluate_and_save()


lst_save = []
for i in range(100,-129):
    name_input = "generated_samples/epoch_126_samples.txt"
    smiles_list = load_smiles_from_txt(name_input)

    metric_evaluator = Metrics(train_smiles_file="cyclic_smiles_only.csv")
    print(name_input)
    results = metric_evaluator.get_metrics(smiles_list)
    print(results)
    lst_save.append([i,results])
