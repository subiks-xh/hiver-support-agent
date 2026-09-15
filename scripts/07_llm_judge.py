import pandas as pd
import os, json, time
from dotenv import load_dotenv
from groq import Groq
import sys
sys.path.insert(0, 'scripts')
from agent_04 import run_agent

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = "openai/gpt-oss-20b"

conv_df = pd.read_csv('data/brand_conversations_english.csv')
sample = conv_df.dropna(subset=['customer_message','brand_reply']).sample(n=30, random_state=42)

print(f"Running LLM judge on 30 examples...\n")

def judge_reply(customer_message, actual_reply, agent_draft):
    prompt = f"""You are evaluating AI-generated Amazon customer support replies.

CUSTOMER MESSAGE:
"{customer_message}"

ACTUAL AMAZON REPLY (ground truth):
"{actual_reply}"

AI AGENT DRAFT:
"{agent_draft}"

Score the AI AGENT DRAFT on these dimensions (1-5):
1. Relevance: Does it address the customer's actual issue?
2. Tone: Is it empathetic and professional?
3. Actionability: Does it offer a clear next step?
4. Accuracy: Is information likely correct? No made-up details?
5. Conciseness: Appropriately brief for Twitter?

Return JSON only:
{{
  "relevance": 4,
  "tone": 4,
  "actionability": 3,
  "accuracy": 4,
  "conciseness": 5,
  "overall_score": 4.0,
  "better_than_actual": false,
  "explanation": "2 sentence explanation"
}}"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = response.choices[0].message.content.strip()
            start = raw.find('{')
            end = raw.rfind('}') + 1
            return json.loads(raw[start:end])
        except Exception as e:
            if '429' in str(e):
                print(f"  Rate limit, waiting 30s...")
                time.sleep(30)
            else:
                print(f"  Error: {str(e)[:60]}")
                return None
    return None

judge_results = []

for i, row in sample.iterrows():
    msg = str(row['customer_message'])
    actual = str(row['brand_reply'])

    print(f"Processing {len(judge_results)+1}/30: {msg[:60]}...")

    try:
        agent_result = run_agent(msg)
        draft = agent_result['draft_reply']
        intent = agent_result['intent']
        time.sleep(2)
    except Exception as e:
        print(f"  Agent error: {e}")
        continue

    scores = judge_reply(msg, actual, draft)
    if scores is None:
        continue

    judge_results.append({
        'customer_message': msg[:200],
        'actual_reply': actual[:200],
        'agent_draft': draft[:200],
        'intent': intent,
        'relevance': scores.get('relevance', 0),
        'tone': scores.get('tone', 0),
        'actionability': scores.get('actionability', 0),
        'accuracy': scores.get('accuracy', 0),
        'conciseness': scores.get('conciseness', 0),
        'overall_score': scores.get('overall_score', 0),
        'better_than_actual': scores.get('better_than_actual', False),
        'explanation': scores.get('explanation', ''),
        'human_score': ''  # Fill manually for 10 examples
    })

    time.sleep(3)

judge_df = pd.DataFrame(judge_results)
os.makedirs('results', exist_ok=True)
judge_df.to_csv('results/judge_results.csv', index=False)

print(f"\n{'='*50}")
print(f"LLM JUDGE RESULTS ({len(judge_df)} examples)")
print(f"{'='*50}")

score_cols = ['relevance','tone','actionability','accuracy','conciseness','overall_score']
for col in score_cols:
    if col in judge_df.columns:
        print(f"  {col:<15}: {judge_df[col].mean():.2f} / 5.0")

better = judge_df['better_than_actual'].sum()
print(f"\n  Agent better than actual reply: {better}/{len(judge_df)} cases")

print(f"\nSample explanations:")
for i, row in judge_df.head(3).iterrows():
    print(f"\n  Customer: {row['customer_message'][:80]}")
    print(f"  Score: {row['overall_score']}")
    print(f"  Explanation: {row['explanation'][:150]}")

# Save 10 for manual scoring
manual = judge_df.head(10)[['customer_message','actual_reply','agent_draft','overall_score','human_score']]
manual.to_csv('results/manual_judge_check.csv', index=False)

print(f"\n✅ Saved judge_results.csv")
print(f"✅ Saved manual_judge_check.csv (fill human_score for 10 examples)")