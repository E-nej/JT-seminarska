from llmlingua import PromptCompressor


# compressor naložimo samo enkrat (prvič počasen zagon, potem hitrejši)
_compressor = None


def get_compressor():
    global _compressor

    if _compressor is None:
        _compressor = PromptCompressor()

    return _compressor


# stisne končni prompt pred pošiljanjem v LLM
def compress_prompt_text(
    prompt: str,  # cel user prompt
    instruction: str = "",  # sistemsko navodilo (lahko prazno)
    question: str = "",  # vhodni stavek, zaradi katerega naj compressor ohrani relevantne informacije
    target_token: int = 1200,
) -> tuple[str, dict]:
    """Returns (compressed_text, stats) where stats has original_tokens, compressed_tokens, ratio."""
    compressor = get_compressor()

    result = compressor.compress_prompt(
        prompt,
        instruction=instruction,
        question=question,
        target_token=target_token,
    )

    original = result.get("origin_tokens") or 0
    compressed = result.get("compressed_tokens") or 0
    stats = {
        "original_tokens": original,
        "compressed_tokens": compressed,
        "ratio": round(compressed / original, 3) if original else None,
    }
    return result["compressed_prompt"], stats
