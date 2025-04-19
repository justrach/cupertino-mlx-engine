import json
from typing import List, Union

import mlx.core as mx
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper
from outlines.models.transformers import TransformerTokenizer
from outlines.processors.structured import JSONLogitsProcessor

from ..schema import ResponseFormat


class OutlinesLogitsProcessor:
    processed_token_count: int = 0

    def __init__(self, tokenizer: TokenizerWrapper, response_format: ResponseFormat):
        json_schema_obj = response_format.json_schema
        schema_dict = json_schema_obj.schema
        self.logits_processor = JSONLogitsProcessor(
            schema_dict,
            TransformerTokenizer(tokenizer._tokenizer),
        )

    def _convert_to_numpy_int(
        self, tokens: Union[mx.array, List[int], None]
    ) -> np.ndarray:
        if tokens is None:
            return np.array([], dtype=np.int32)

        if isinstance(tokens, (list, mx.array)):
            tokens_data = tokens.tolist() if isinstance(tokens, mx.array) else tokens
            return np.asarray(tokens_data, dtype=np.int32)

        return np.array([], dtype=np.int32)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        generated_tokens = (
            self._convert_to_numpy_int(tokens[-self.processed_token_count :])
            if self.processed_token_count > 0
            else np.array([], dtype=np.int32)
        )

        logits_data = (
            logits.astype(mx.float32) if logits.dtype == mx.bfloat16 else logits
        )
        logits_flat = logits_data.reshape(-1)

        logits_np = np.asarray(logits_flat.tolist(), dtype=np.float32)
        processed_logits = self.logits_processor(generated_tokens, logits_np)

        self.processed_token_count += 1
        return mx.array(processed_logits).reshape(1, -1)
