import numpy as np

from ._SubspaceMotif import SubspaceMotif


class SubspaceMotifSet:

    _motifs: list[SubspaceMotif]

    def __init__(self, motifs: list[SubspaceMotif], sort: bool = True):
        self._motifs = sorted(motifs) if sort else motifs

    @staticmethod
    def from_arrays(
        arrays: (np.array, np.ndarray), sort: bool = True
    ) -> "SubspaceMotifSet":
        return SubspaceMotifSet(
            motifs=[SubspaceMotif(arrays[0], indices) for indices in arrays[1]],
            sort=sort,
        )

    @property
    def to_arrays(self) -> (np.array, np.ndarray):
        return self.mask, np.array([motif.to_arrays[1] for motif in self])

    @staticmethod
    def from_motifs(
        motifs: list[SubspaceMotif], sort: bool = True
    ) -> "SubspaceMotifSet":
        return SubspaceMotifSet(motifs=motifs, sort=sort)

    @property
    def to_motifs(self) -> list[SubspaceMotif]:
        return self._motifs

    @property
    def mask(self) -> np.array:
        return self._motifs[0].mask

    def motif(self, i: int) -> SubspaceMotif:
        return self._motifs[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item) -> SubspaceMotif:
        return self.motif(item)

    def __len__(self) -> int:
        return len(self._motifs)
