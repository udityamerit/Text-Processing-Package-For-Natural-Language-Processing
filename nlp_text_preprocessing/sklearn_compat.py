"""scikit-learn compatible transformer (needs the optional ``[sklearn]`` extra)."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError as exc:  # pragma: no cover - exercised only without scikit-learn
    raise ImportError(
        "TextPreprocessor needs scikit-learn. Install it with: "
        "pip install \"nlp-text-preprocessing[sklearn]\""
    ) from exc

from .all_text_preprocess import Pipeline


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Use text preprocessing steps inside a scikit-learn pipeline.

    Parameters
    ----------
    steps : list, optional
        Same step format as :class:`nlp_text_preprocessing.Pipeline`. Defaults
        to ``Pipeline.default()`` when omitted.

    Example
    -------
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> model = make_pipeline(TextPreprocessor(), TfidfVectorizer())
    """

    def __init__(self, steps: Optional[Sequence[Any]] = None) -> None:
        self.steps = steps

    def _pipeline(self):
        return Pipeline(self.steps) if self.steps is not None else Pipeline.default()

    def fit(self, X: Any = None, y: Any = None) -> "TextPreprocessor":
        self._pipeline()          # validate the steps
        return self

    def transform(self, X: Iterable[Any]) -> List[Any]:
        return self._pipeline().run_batch(X)

    def __sklearn_is_fitted__(self) -> bool:
        return True
