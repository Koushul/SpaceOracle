from glob import glob
import anndata
import scanpy as sc
import warnings

# Suppress ImplicitModificationWarning
warnings.simplefilter(action='ignore', category=anndata.ImplicitModificationWarning)

def load_example_slideseq(path_dir):
    """Load an example SlideSeq dataset."""
    return [(i, anndata.read_h5ad(i)) for i in glob(path_dir + '/*.h5ad')]


def filter_adata(adata, min_counts=300, min_cells=10, n_top_genes=2000):
    """Filter anndata object."""
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    adata = adata[:, ~adata.var["mt"]]
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    adata.layers["raw_count"] = adata.X
    
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    
    adata = adata[:, adata.var.highly_variable]
    
    return adata