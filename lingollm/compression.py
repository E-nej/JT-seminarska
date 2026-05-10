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
) -> str:

    compressor = get_compressor()

    result = compressor.compress_prompt(
        prompt,
        instruction=instruction,
        question=question,
        target_token=target_token,
    )

    return result["compressed_prompt"]
