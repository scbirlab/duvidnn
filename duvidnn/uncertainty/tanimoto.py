
from dataclasses import dataclass

from aspect import DataPipeline
import torch


@dataclass
class Tanimoto:
    column: str = "fingerprint"
    structure_column: str = "smiles"
    fingerprint_transform: str = "morgan-fingerprint"

    name = "tanimoto"

    def _fingerprint_pipeline(self):
        return DataPipeline({
            self.column: (
                self.structure_column,
                self.fingerprint_transform,
            ),
        })

    def _add_fingerprint(self, data):
        if self.column in data.column_names:
            return data
        if self.structure_column not in data.column_names:
            raise ValueError(
                f"Tanimoto requires fingerprint column "
                f"{self.column!r} or structure column {self.structure_column}, but training data have "
                f"{data.column_names}."
            )
        pipeline = self._fingerprint_pipeline()
        source = data.select_columns([self.structure_column])
        fingerprinted = pipeline(source)
        fingerprints = (
            fingerprinted
            .with_format(None)
            [self.column]
        )
        return data.add_column(
            self.column,
            fingerprints
        )

    def prepare_candidates(self, data):
        return self._add_fingerprint(data)
        
    def prepare(
        self,
        box,
        **kwargs
    ):
        training_data = box._training_data()
        training_data = self._add_fingerprint(training_data)
        return training_data.with_format(None)[self.column]

    def __call__(
        self,
        *,
        box=None,
        batch,
        prediction=None,
        state
    ):
        if self.column not in batch:
            raise ValueError(
                f"Tanimoto requires fingerprint column "
                f"{self.column!r} in candidate data."
            )

        query = torch.as_tensor(
            batch[self.column],
            dtype=torch.float32,
        )

        references = torch.as_tensor(
            state,
            device=query.device,
            dtype=query.dtype,
        )

        intersection = (
            query
            .unsqueeze(1)
            .mul(references.unsqueeze(0))
            .sum(dim=-1)
        )

        query_total = query.sum(dim=-1, keepdim=True)
        ref_total = references.sum(dim=-1).unsqueeze(0)
        union = query_total + ref_total - intersection

        eps = torch.finfo(query.dtype).eps
        similarity = intersection / union.clamp_min(eps)
        return similarity.max(dim=-1).values


