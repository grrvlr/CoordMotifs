import numpy as np

from ._sub_tsmd import _overlap_rate


class SubspaceMotif:
    _mask: np.array
    _indices: np.ndarray
    _subspace_index: np.array

    def __init__(self, mask: np.array, indices: np.ndarray):
        self._mask = mask
        self._indices = indices
        self._subspace_index = mask.cumsum() - 1

    def __lt__(self, other: "SubspaceMotif") -> bool:
        if self.start == other.start:
            return self.end() < other.end()
        else:
            return self.start() < other.start()

    @staticmethod
    def from_arrays(mask: np.array, indices: np.ndarray) -> "SubspaceMotif":
        return SubspaceMotif(mask=mask, indices=indices)

    @property
    def to_arrays(self) -> (np.array, np.ndarray):
        return self.mask, self._indices

    @staticmethod
    def from_dict(
        dictionary: dict[int, (float, float)], dimension: int
    ) -> "SubspaceMotif":
        mask = np.zeros(shape=dimension, dtype=bool)
        indices = np.empty(shape=(2, len(dictionary)))
        for i, (d, interval) in enumerate(dictionary.items()):
            mask[d] = True
            indices[:, i] = interval
        return SubspaceMotif(mask=mask, indices=indices)

    @property
    def to_dict(self) -> (dict[int, (float, float)], int):
        dictionary = {
            attribute: self.on_attribute(attribute) for attribute in self.subspace
        }
        return dictionary, self.dimension

    @property
    def mask(self) -> np.array:
        return self._mask

    @property
    def dimension(self) -> int:
        return self.mask.sum()

    @property
    def subspace(self) -> np.array:
        return np.where(self._mask)[0]

    def start(self, subspace: int = None) -> float:
        if subspace is None:
            return self._indices.min()
        else:
            return self.on_attribute(subspace).min()

    def end(self, subspace: int = None) -> float:
        if subspace is None:
            return self._indices.max()
        else:
            return self.on_attribute(subspace).max()

    def length(self, subspace: int = None) -> float:
        return self.end(subspace) - self.start(subspace)

    def on_attribute(self, attribute: int) -> np.array:
        if self._mask[attribute]:
            return self._indices[:, self._subspace_index[attribute]]
        else:
            return ValueError(
                f"Invalid attribute given: '{attribute}'. Valid options are {self.subspace}"
            )

    def overlap_rate(
        self, other: "SubspaceMotif", linkage: str = "average"
    ) -> np.floating:
        pairwise_overlap_rates = [
            _overlap_rate(
                self.start(subspace),
                self.end(subspace),
                other.start(subspace_other),
                other.end(subspace_other),
            )
            for subspace in self.subspace
            for subspace_other in other.subspace
        ]
        if linkage == "average":
            return np.mean(pairwise_overlap_rates)
        if linkage == "complete":
            # min instead of max because overlap rate is a similarity and not a distance
            return np.min(pairwise_overlap_rates)
        raise ValueError(
            f"Invalid linkage given: '{linkage}'. Valid options are ['average', 'complete']"
        )
