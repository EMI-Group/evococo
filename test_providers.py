import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import sys

# Load EvoCoder config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.config import LLM_PROVIDERS

async def test_provider(provider_name, config):
    print(f"\n--- Testing [{provider_name.upper()}] ---")
    api_key = os.getenv(config['api_key_env'])
    
    if not api_key or api_key.startswith("your_"):
        print(f"❌ SKIPPED: Valid API key not found in {config['api_key_env']}")
        return False
        
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config['base_url'],
        timeout=60.0,
        max_retries=1
    )
    
    try:
        print(f"Connecting to {config['base_url']} using model '{config['model']}'...")
        response = await client.chat.completions.create(
            model=config['model'],
            messages=[{"role": "user", "content": "Reply with exactly one word: 'OK'"}],
            temperature=0.1,
            max_tokens=100
        )
        content = response.choices[0].message.content
        if content is None:
            print(f"❌ ERROR: Received None. Full response: {response}")
            return False
        reply = content.strip()
        print(f"✅ SUCCESS: Received response -> {reply}")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

async def main():
    load_dotenv()
    print("Starting LLM Provider Connectivity Test...")
    
    results = {}
    for name, config in LLM_PROVIDERS.items():
        if name != "litellm": continue
        res = await test_provider(name, config)
        results[name] = res
        
    print("\n=== SUMMARY ===")
    for name, res in results.items():
        status = "✅ PASS" if res else "❌ FAIL"
        print(f"{name.ljust(10)}: {status}")

if __name__ == "__main__":
    asyncio.run(main())
