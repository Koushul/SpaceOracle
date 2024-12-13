import os
import pandas as pd
import numpy as np
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tqdm import tqdm


@dataclass
class BetaOutput:
    betas: np.ndarray
    modulator_genes: List[str]
    modulator_gene_indices: List[int]
    ligands: Optional[List[str]] = None
    receptors: Optional[List[str]] = None
    tfl_ligands: Optional[List[str]] = None
    tfl_regulators: Optional[List[str]] = None
    ligand_receptor_pairs: Optional[List[Tuple[str, str]]] = None
    tfl_pairs: Optional[List[Tuple[str, str]]] = None
    wbetas: Optional[Tuple[str, pd.DataFrame]] = None


class BetaFrame(pd.DataFrame):

    @classmethod
    def from_path(cls, path):
        df = pd.read_parquet(path)
        return cls(df)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prefix = 'beta_'
        self.tfs = []
        self.lr_pairs = []
        self.tfl_pairs = []

        # to be filled in later
        self.modulator_gene_indices = None
        self.wbetas = None
        
        for col in self.columns:
            if col.startswith(self.prefix):
                modulator = col[len(self.prefix):]
                if '$' in modulator:
                    self.lr_pairs.append(modulator)
                elif '#' in modulator:
                    self.tfl_pairs.append(modulator) 
                else:
                    self.tfs.append(modulator)


        self.ligands, self.receptors = zip(
            *[p.split('$') for p in self.lr_pairs]) if self.lr_pairs else ([], [])
        self.tfl_ligands, self.tfl_regulators = zip(
            *[p.split('#') for p in self.tfl_pairs]) if self.tfl_pairs else ([], [])
        self.ligands = list(self.ligands)
        self.receptors = list(self.receptors)
        self.tfl_ligands = list(self.tfl_ligands)
        self.tfl_regulators = list(self.tfl_regulators)
        self.modulators_genes = [f'beta_{m}' for m in np.unique(
                self.tfs + self.ligands + self.receptors + \
                self.tfl_ligands + self.tfl_regulators)
            ]
        
        self._all_ligands = np.unique(list(self.ligands) + list(self.tfl_ligands))

        self.df_lr_columns = [f'beta_{r}' for r in self.receptors]+ \
            [f'beta_{l}' for l in self.ligands]
        self.df_tfl_columns = [f'beta_{r}' for r in self.tfl_regulators]+ \
            [f'beta_{l}' for l in self.tfl_ligands]
        self.tf_columns = [f'beta_{t}' for t in self.tfs]

        self.lr_pairs = [pair.split('$') for pair in self.lr_pairs]
        self.tfl_pairs = [pair.split('#') for pair in self.tfl_pairs]
    

    def splash(self, rw_ligands, gex_df):
        _df = pd.DataFrame(
            np.concatenate([
                self[self.tf_columns].to_numpy(),
                self[[f'beta_{a}${b}' for a, b in zip(self.ligands, self.receptors)]*2].to_numpy() * \
                    rw_ligands[self.ligands].join(gex_df[self.receptors]).to_numpy(),
                self[[f'beta_{a}#{b}' for a, b in zip(self.tfl_ligands, self.tfl_regulators)]*2].to_numpy() * \
                    rw_ligands[self.tfl_ligands].join(gex_df[self.tfl_regulators]).to_numpy()
            ], axis=1),
            index=self.index,
            columns=self.tf_columns + self.df_lr_columns + self.df_tfl_columns
        ).groupby(lambda x: x, axis=1).sum()

        return _df[self.modulators_genes]
        
    def _repr_html_(self):
        info = f"BetaFrame with {len(self.modulators_genes)} modulator genes\n"
        df_html = super()._repr_html_()
        return f"<div><p>{info}</p>{df_html}</div>"


class Betabase:
    """
    Holds a collection of BetaFrames for each gene.
    """
    def __init__(self, adata, folder):
        assert os.path.exists(folder), f'Folder {folder} does not exist'
        # self.adata = adata
        self.xydf = pd.DataFrame(
            adata.obsm['spatial'], index=adata.obs_names)
        self.folder = folder
        self.gene2index = dict(
            zip(
                adata.var_names, 
                range(len(adata.var_names))
            )
        )
        self.beta_paths = glob.glob(f'{self.folder}/*_betadata.parquet')

        self.data = {}
        self.ligands_set = set()

        self.load_betas_from_disk()

    def __len__(self):
        return len(self.data)


    def load_betas_from_disk(self):
        for path in tqdm(self.beta_paths):
            gene_name = path.split('/')[-1].split('_')[0]
            self.data[gene_name] = BetaFrame.from_path(path)
            self.ligands_set.update(self.data[gene_name]._all_ligands)
        
        for gene_name, betadata in self.data.items():
            betadata.modulator_gene_indices = [
                self.gene2index[g.replace('beta_', '')] for g in betadata.modulators_genes
            ]
