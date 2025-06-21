from typing import Callable, Tuple, List, Any, TypeVar, Generic, Optional
from dataclasses import dataclass
import pandas as pd

T = TypeVar('T')  # Generic search space element type

@dataclass
class RecursiveInfo(Generic[T]):
    max_overall: float
    max_overall_slice: List[T]
    max_prefix: float
    max_prefix_slice: List[T]
    max_suffix: float
    max_suffix_slice: List[T]
    full_slice: List[T]

    def __repr__(self):
        return (f"RecursiveInfo(max_overall={self.max_overall:.4f}, "
                f"max_prefix={self.max_prefix:.4f}, max_suffix={self.max_suffix:.4f})")


def generalized_max_substructure(
    search_space: List[T],
    partition_fn: Callable[[List[T]], Tuple[List[T], List[T]]],
    base_case_fn: Callable[[List[T]], bool],
    base_case_eval_fn: Callable[[List[T]], RecursiveInfo[T]],
    combine_fn: Callable[[RecursiveInfo[T], RecursiveInfo[T]], RecursiveInfo[T]]
) -> RecursiveInfo[T]:
    """
    Generalized divide-and-conquer solver for contiguous substructure optimization.
    """
    if base_case_fn(search_space):
        return base_case_eval_fn(search_space)

    left, right = partition_fn(search_space)
    left_info = generalized_max_substructure(left, partition_fn, base_case_fn, base_case_eval_fn, combine_fn)
    right_info = generalized_max_substructure(right, partition_fn, base_case_fn, base_case_eval_fn, combine_fn)

    return combine_fn(left_info, right_info)

