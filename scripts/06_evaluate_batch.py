import pandas as pd
import os, json, time
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = "openai/gpt-oss-20b"

with open('data/intents.json', 'r') as f:
    INTENTS = json.load(f)
INTENT_LIST = [i['intent'] for i in INTENTS]

golden_df = pd.read_csv('data/golden_set_verified.csv')
print(f"Loaded {len(golden_df)} examples")
print(f"\nIntent distribution:")
print(golden_df['human_intent'].value_counts())

# ============================================================
# BASELINES
# ============================================================
most_common = golden_df['human_intent'].value_counts().index[0]

KEYWORD_MAP = {
    'ORDER_STATUS': ['order', 'track', 'tracking', 'shipped', 'shipping',
                     'arrive', 'package', 'where is my', 'when will', 'status'],
    'DELIVERY_ISSUE': ['wrong address', 'not delivered', 'left outside',
                       'delivered to wrong', 'never arrived', 'lost package',
                       'not received', 'failed delivery', 'missing parcel'],
    'RETURN_REFUND': ['return', 'refund', 'money back', 'send back',
                      'reimburse', 'exchange', 'replace', 'cancel order'],
    'PAYMENT_ISSUE': ['charged', 'charge', 'billing', 'payment', 'paid',
                      'fee', 'invoice', 'double charged', 'overcharged'],
    'TECHNICAL_ISSUE': ['app', 'website', 'login', 'sign in', 'password',
                        'error', 'not working', 'kindle', 'fire tv', 'alexa',
                        'prime video', 'streaming', 'loading', 'bug'],
    'PRODUCT_ISSUE': ['broken product', 'damaged', 'defective', 'wrong item',
                      'missing parts', 'fake', 'not as described'],
    'CUSTOMER_COMPLAINT': ['terrible', 'worst', 'horrible', 'unacceptable',
                           'pathetic', 'disgusting', 'furious', 'awful',
                           'useless', 'sue', 'lawyer', 'legal', 'fraud', 'scam']
}

def baseline_trivial(msg): return most_common

def baseline_keyword(msg):
    msg_lower = str(msg).lower()
    scores = {intent: sum(1 for k in kws if k in msg_lower)
              for intent, kws in KEYWORD_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else most_common

# ============================================================
# BATCH CLASSIFY - Send 10 messages per API call
# ============================================================
def batch_classify(messages, batch_size=10):
    """Classify multiple messages in one API call to save rate limit"""
    results = []

    for batch_start in range(0, len(messages), batch_size):
        batch = messages[batch_start: batch_start + batch_size]

        numbered = "\n".join([f"{j+1}. {str(msg)[:200]}"
                              for j, msg in enumerate(batch)])

        prompt = f"""Classify each Amazon customer support message into exactly one intent.

Intents:
- ORDER_STATUS: tracking/delivery status questions
- DELIVERY_ISSUE: wrong address, not delivered, missing package  
- RETURN_REFUND: returns, refunds, money back
- PAYMENT_ISSUE: charges, billing, payment problems
- TECHNICAL_ISSUE: app, device, website problems
- PRODUCT_ISSUE: broken, wrong, defective item
- CUSTOMER_COMPLAINT: angry, threats, legal action

Messages:
{numbered}

Return a JSON array with one object per message (same order):
[
  {{"id": 1, "intent": "ORDER_STATUS"}},
  {{"id": 2, "intent": "DELIVERY_ISSUE"}}
]

Return ONLY the JSON array."""

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                raw = response.choices[0].message.content.strip()
                start = raw.find('[')
                end = raw.rfind(']') + 1
                batch_results = json.loads(raw[start:end])

                for item in batch_results:
                    intent = item.get('intent', 'ORDER_STATUS')
                    if intent not in INTENT_LIST:
                        intent = 'ORDER_STATUS'
                    results.append(intent)

                print(f"  ✅ Batch {batch_start//batch_size + 1}: classified {len(batch)} messages")
                time.sleep(3)
                break

            except Exception as e:
                if '429' in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"  ⏳ Rate limit, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  ❌ Error: {str(e)[:60]}")
                    # Fallback: keyword for this batch
                    for msg in batch:
                        results.append(baseline_keyword(msg))
                    break
        else:
            # All 3 attempts failed
            print(f"  ❌ Batch failed, using keyword fallback")
            for msg in batch:
                results.append(baseline_keyword(msg))

    return results

# ============================================================
# RUN EVALUATION
# ============================================================
print(f"\nClassifying {len(golden_df)} messages in batches of 10...")
print("(Much faster - only 19 API calls instead of 189!)\n")

messages = golden_df['customer_message'].tolist()
agent_predictions = batch_classify(messages, batch_size=10)

print(f"\nGot {len(agent_predictions)} predictions")

# Build results
results = []
for i, row in golden_df.iterrows():
    msg = str(row['customer_message'])
    true_intent = str(row['human_intent'])
    agent_intent = agent_predictions[i] if i < len(agent_predictions) else most_common

    results.append({
        'message': msg[:150],
        'true_intent': true_intent,
        'agent_intent': agent_intent,
        'trivial_intent': baseline_trivial(msg),
        'keyword_intent': baseline_keyword(msg),
        'agent_correct': agent_intent == true_intent,
        'keyword_correct': baseline_keyword(msg) == true_intent,
        'trivial_correct': baseline_trivial(msg) == true_intent,
    })

results_df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/evaluation_results.csv', index=False)

# ============================================================
# METRICS
# ============================================================
print("\n" + "="*60)
print("FINAL EVALUATION RESULTS")
print("="*60)

agent_acc   = accuracy_score(results_df['true_intent'], results_df['agent_intent'])
trivial_acc = accuracy_score(results_df['true_intent'], results_df['trivial_intent'])
keyword_acc = accuracy_score(results_df['true_intent'], results_df['keyword_intent'])

print(f"\n{'System':<40} {'Accuracy':>10}")
print("-" * 52)
print(f"{'Trivial Baseline (always '+most_common+')':<40} {trivial_acc:>10.1%}")
print(f"{'Keyword Baseline':<40} {keyword_acc:>10.1%}")
print(f"{'Our Agent (LLM Batch)':<40} {agent_acc:>10.1%}")
print(f"\n  Improvement over trivial:  +{(agent_acc-trivial_acc)*100:.1f}%")
print(f"  Improvement over keyword:  +{(agent_acc-keyword_acc)*100:.1f}%")

print("\n--- Agent Per-Intent Report ---")
print(classification_report(
    results_df['true_intent'],
    results_df['agent_intent'],
    zero_division=0
))

print("\n--- Keyword Baseline Report ---")
print(classification_report(
    results_df['true_intent'],
    results_df['keyword_intent'],
    zero_division=0
))

# Save metrics
metrics = {
    'n_examples': len(results_df),
    'agent_accuracy': round(agent_acc, 4),
    'trivial_baseline_accuracy': round(trivial_acc, 4),
    'keyword_baseline_accuracy': round(keyword_acc, 4),
    'improvement_over_trivial': round(agent_acc - trivial_acc, 4),
    'improvement_over_keyword': round(agent_acc - keyword_acc, 4),
    'most_common_intent': most_common,
    'per_intent_agent': results_df.groupby('true_intent')['agent_correct'].mean().round(3).to_dict(),
    'per_intent_keyword': results_df.groupby('true_intent')['keyword_correct'].mean().round(3).to_dict(),
}
with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Saved metrics.json and evaluation_results.csv")
print(f"\n🎯 HEADLINE NUMBER: Agent Accuracy = {agent_acc:.1%}")