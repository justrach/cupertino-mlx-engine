from typing import List, NamedTuple, Optional, Union

from mlx_lm.tokenizer_utils import TokenizerWrapper


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


class StopTokensChecker:
    """A class to handle custom stop sequence checks."""

    def __init__(
        self,
        stop_words: Optional[Union[str, List[str]]],
        tokenizer: TokenizerWrapper,
    ):
        """Initialize the TokenStopChecker.

        Args:
            stop_words: Words or phrases that should trigger stopping
            tokenizer: The tokenizer to use for encoding stop words
        """
        self._stop_id_sequences = self._prepare_stop_sequences(stop_words, tokenizer)

    def _prepare_stop_sequences(
        self, stop_words: Optional[Union[str, List[str]]], tokenizer: TokenizerWrapper
    ) -> List[List[int]]:
        """Prepare stop sequences by converting words to token IDs.

        Args:
            stop_words: Words to convert to token sequences
            tokenizer: Tokenizer to use for encoding

        Returns:
            List of token ID sequences
        """
        if not stop_words:
            return []

        words = [stop_words] if isinstance(stop_words, str) else stop_words
        return [
            tokenizer.encode(word, add_special_tokens=False)
            for word in words
            if word  # Skip empty strings
        ]

    def check_stop_condition(self, tokens: List[int]) -> StopCondition:
        """Check if the token sequence meets any custom stop criteria.

        Args:
            tokens: The current sequence of generated tokens

        Returns:
            StopCondition indicating if/how generation should stop
        """
        if not tokens or not self._stop_id_sequences:
            return StopCondition(stop_met=False, trim_length=0)

        # Check stop sequences
        for stop_ids in self._stop_id_sequences:
            stop_len = len(stop_ids)
            if len(tokens) >= stop_len and tokens[-stop_len:] == stop_ids:
                # Look for partial matches to trim
                prefix_len = self._find_prefix_length(tokens[:-stop_len], stop_ids[0])
                return StopCondition(stop_met=True, trim_length=stop_len + prefix_len)

        return StopCondition(stop_met=False, trim_length=0)

    @staticmethod
    def _find_prefix_length(tokens: List[int], first_stop_token: int) -> int:
        """Find length of matching prefix tokens.

        Args:
            tokens: Token sequence to check
            first_stop_token: First token of stop sequence

        Returns:
            Length of matching prefix
        """
        prefix_len = 0
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == first_stop_token:
                prefix_len += 1
            else:
                break
        return prefix_len
