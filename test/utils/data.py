from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem

def _cached_molgraph(smiles: str) -> dict:
    """Create an Arrow/HF-like serialized Chemprop MolGraph."""
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    molgraph = featurizer(
        Chem.MolFromSmiles(smiles)
    )

    return {
        "V": molgraph.V.tolist(),
        "E": molgraph.E.tolist(),
        "edge_index": molgraph.edge_index.tolist(),
        "rev_edge_index": molgraph.rev_edge_index.tolist(),
    }


def _make_vectome_rows():
    import torch
    return torch.tensor([
            [.1, .2, .3, .4],
            [.5, .6, .7, .8],
    ], dtype=torch.float32)
    

def _make_chemprop_rows(smiles=("CCO", "CCN")):
    return [{
            "mg": _cached_molgraph(s),
            "V_d": None,
            "X_d": None,
    } for s in smiles]
