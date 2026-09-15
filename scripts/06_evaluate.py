import pandas as pd
import os, json, sys, time
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = "openai/gpt-oss-20b"

sys.path.insert(0, 'scripts')
from agent_04 import classify_intent

with open('data/intents.json', 'r') as f:
    INTENTS = json.load(f)
INTENT_LIST = [i['intent'] for i in INTENTS]

# Load golden set
golden_df = pd.read_csv('data/golden_set_verified.csv')
print(f"Loaded {len(golden_df)} verified examples")
print(f"\nIntent distribution:")
print(golden_df['human_intent'].value_counts())

# ============================================================
# BASELINE 1: TRIVIAL
# ============================================================
most_common = golden_df['human_intent'].value_counts().index[0]
print(f"\nTrivial baseline always predicts: {most_common}")

def baseline_trivial(message):
    return most_common

# ============================================================
# BASELINE 2: KEYWORD
# ============================================================
KEYWORD_MAP = {
    'ORDER_STATUS': [
        'order', 'track', 'tracking', 'delivery status', 'shipped',
        'shipping', 'arrive', 'package', 'where is my', 'when will',
        'dispatch', 'status', 'expected delivery'
    ],
    'DELIVERY_ISSUE': [
        'wrong address', 'not delivered', 'left outside', 'neighbour',
        'neighbor', 'delivered to wrong', 'never arrived', 'lost package',
        'not received', 'delivery attempt', 'failed delivery', 'missing parcel'
    ],
    'RETURN_REFUND': [
        'return', 'refund', 'money back', 'send back', 'reimburse',
        'reimbursement', 'exchange', 'replace', 'replacement',
        'cancel order', 'cancelled', 'give back'
    ],
    'PAYMENT_ISSUE': [
        'charged', 'charge', 'billing', 'payment', 'paid',
        'price', 'fee', 'invoice', 'receipt', 'transaction',
        'double charged', 'overcharged', 'unauthorized charge'
    ],
    'TECHNICAL_ISSUE': [
        'app', 'website', 'login', 'sign in', 'password', 'error',
        'not working', 'kindle', 'fire tv', 'alexa', 'prime video',
        'streaming', 'loading', 'bug', 'crash', 'account access'
    ],
    'PRODUCT_ISSUE': [
        'broken product', 'damaged', 'defective', 'wrong item',
        'wrong product', 'missing parts', 'incomplete', 'fake',
        'counterfeit', 'not as described', 'poor quality'
    ],
    'CUSTOMER_COMPLAINT': [
        'terrible', 'worst', 'horrible', 'unacceptable', 'disgrace',
        'pathetic', 'disgusting', 'outrageous', 'furious', 'awful',
        'useless', 'incompetent', 'disappointed', 'sue', 'lawyer',
        'legal', 'fraud', 'scam', 'never again', 'shocking'
    ]
}

def baseline_keyword(message):
    message_lower = str(message).lower()
    scores = {intent: 0 for intent in KEYWORD_MAP}
    for intent, keywords in KEYWORD_MAP.items():
        scores[intent] = sum(1 for k in keywords if k in message_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else most_common

# ============================================================
# RUN EVALUATION
# ============================================================
print("\nRunning evaluation...")
print("(Agent API calls - takes 3-4 minutes)\n")

results = []
total = len(golden_df)

for i, row in golden_df.iterrows():
    msg = str(row['customer_message'])
    true_intent = str(row['human_intent'])

    # Baselines (instant, no API)
    trivial_pred = baseline_trivial(msg)
    keyword_pred = baseline_keyword(msg)

    # Agent prediction
    try:
        agent_result = classify_intent(msg)
        agent_intent = agent_result['intent']
        agent_conf = agent_result.get('confidence', 0.5)
        time.sleep(1.5)  # Avoid rate limit
    except Exception as e:
        if '429' in str(e):
            print(f"  Rate limit at row {i}, waiting 20s...")
            time.sleep(20)
            try:
                agent_result = classify_intent(msg)
                agent_intent = agent_result['intent']
                agent_conf = agent_result.get('confidence', 0.5)
            except:
                agent_intent = keyword_pred
                agent_conf = 0.0
        else:
            agent_intent = keyword_pred
            agent_conf = 0.0

    results.append({
        'message': msg[:150],
        'true_intent': true_intent,
        'agent_intent': agent_intent,
        'agent_confidence': agent_conf,
        'trivial_intent': trivial_pred,
        'keyword_intent': keyword_pred,
        'agent_correct': agent_intent == true_intent,
        'keyword_correct': keyword_pred == true_intent,
        'trivial_correct': trivial_pred == true_intent,
    })

    if (i + 1) % 20 == 0:
        done = i + 1
        agent_acc_so_far = sum(r['agent_correct'] for r in results) / len(results)
        print(f"  Progress: {done}/{total} | Agent acc so far: {agent_acc_so_far:.1%}")

results_df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/evaluation_results.csv', index=False)

# ============================================================
# PRINT METRICS
# ============================================================
print("\n" + "="*60)
print("FINAL EVALUATION RESULTS")
print("="*60)

agent_acc   = accuracy_score(results_df['true_intent'], results_df['agent_intent'])
trivial_acc = accuracy_score(results_df['true_intent'], results_df['trivial_intent'])
keyword_acc = accuracy_score(results_df['true_intent'], results_df['keyword_intent'])

print(f"\n{'System':<35} {'Accuracy':>10}")
print("-" * 47)
print(f"{'Trivial Baseline (always ' + most_common + ')':<35} {trivial_acc:>10.1%}")
print(f"{'Keyword Baseline':<35} {keyword_acc:>10.1%}")
print(f"{'Our Agent (LLM)':<35} {agent_acc:>10.1%}")
print(f"\n  Improvement over trivial:  +{(agent_acc-trivial_acc)*100:.1f}%")
print(f"  Improvement over keyword:  +{(agent_acc-keyword_acc)*100:.1f}%")

print("\n--- Agent Per-Intent Report ---")
print(classification_report(
    results_df['true_intent'],
    results_df['agent_intent'],
    zero_division=0
))

print("\n--- Keyword Baseline Per-Intent Report ---")
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
    'per_intent_agent_accuracy': results_df.groupby('true_intent')['agent_correct'].mean().round(3).to_dict(),
    'per_intent_keyword_accuracy': results_df.groupby('true_intent')['keyword_correct'].mean().round(3).to_dict(),
}

with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Saved metrics to results/metrics.json")
print(f"✅ Saved results to results/evaluation_results.csv")
print(f"\n🎯 HEADLINE NUMBER: Agent Accuracy = {agent_acc:.1%}")
