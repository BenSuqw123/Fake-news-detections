import asyncio
from user_input_processing.evaluation_layer import evaluate_final

async def test():
    print("Testing Groq migration...")

    result = await evaluate_final(
        claim="Công dân Việt Nam có quyền bầu cử khi đủ 18 tuổi",
        docs=[{
            "text": "Điều 27. Công dân đủ mười tám tuổi có quyền bầu cử.",
            "metadata": {"law_title": "Hiến pháp 2013", "article": "27"},
            "rerank_score": 0.95,
            "score": 0.95
        }],
        contra_docs=[]
    )

    print(f"Verdict:    {result['verdict']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning:  {result['reasoning'][:100]}")

    assert result["verdict"] == "SUPPORTED", \
        f"Expected SUPPORTED, got {result['verdict']}"
    assert result["confidence"] > 0.5, \
        f"Confidence too low: {result['confidence']}"

    print("✓ Migration successful!")

asyncio.run(test())
