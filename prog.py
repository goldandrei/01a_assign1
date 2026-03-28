import os
import time
import pandas as pd
import httpx
import asyncio

# כמה מוצרים להריץ
NUM_ROWS = 10

# טעינת הדאטה מקובץ Excel
df = pd.read_excel("product_dataset.xlsx")


def build_product_input(row: pd.Series) -> str:
    return f"""
Product name: {row.get('product_name', '')}
Product attributes: {row.get('Product_attribute_list', '')}
Material: {row.get('material', '')}
Warranty: {row.get('warranty', '')}
""".strip()


SYSTEM_PROMPT = """
You are an assistant that writes product descriptions for an e-commerce website.

Generate a persuasive, concise, and clear product description using only the provided product data.

Rules:
- Write 50-90 words total
- Use a friendly, credible sales tone
- Use only the provided information
- Do not invent features, specifications, dimensions, or benefits
- Keep the text natural, easy to read, and suitable for an online store
- Return only the product description text

Important:
- Ground every claim in the provided product data
- If some information is missing, do not guess
""".strip()


NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "").strip()
NEBIUS_CHAT_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-fast"

if not NEBIUS_API_KEY:
    raise ValueError("NEBIUS_API_KEY is not set")


async def generate_description(product_input: str) -> dict:
    headers = {
        "Authorization": f"Bearer {NEBIUS_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": product_input},
        ],
        "temperature": 0.2,
    }

    start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            NEBIUS_CHAT_URL,
            headers=headers,
            json=payload,
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    if response.status_code >= 400:
        raise RuntimeError(
            f"Nebius API error: {response.status_code} - {response.text[:300]}"
        )

    data = response.json()

    content = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})

    return {
        "generated_description": content,
        "latency_ms": latency_ms,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


async def main():
    results = []

    for i in range(min(NUM_ROWS, len(df))):
        row = df.iloc[i]
        product_input = build_product_input(row)

        try:
            result = await generate_description(product_input)

            results.append({
                "product_name": row.get("product_name", ""),
                "Product_attribute_list": row.get("Product_attribute_list", ""),
                "material": row.get("material", ""),
                "warranty": row.get("warranty", ""),
                "generated_description": result["generated_description"],
                "latency_ms": result["latency_ms"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            })

            print(f"Done: row {i}")

        except Exception as e:
            print(f"Error on row {i}: {e}")

            results.append({
                "product_name": row.get("product_name", ""),
                "Product_attribute_list": row.get("Product_attribute_list", ""),
                "material": row.get("material", ""),
                "warranty": row.get("warranty", ""),
                "generated_description": "",
                "latency_ms": None,
                "input_tokens": None,
                "output_tokens": None,
            })

    results_df = pd.DataFrame(results)

    print("\n=== RESULTS ===")
    print(results_df.head())

    results_df.to_excel("assignment_01_results.xlsx", index=False)
    print("\nSaved to assignment_01_results.xlsx")


if __name__ == "__main__":
    asyncio.run(main())