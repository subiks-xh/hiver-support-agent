import pandas as pd
import json
import os

# Load results
results_df = pd.read_csv('results/evaluation_results.csv')
judge_df = pd.read_csv('results/judge_results.csv')

print("="*60)
print("FAILURE ANALYSIS REPORT")
print("="*60)

# ============================================================
# INTENT MISCLASSIFICATIONS
# ============================================================
failures = results_df[results_df['agent_correct'] == False].copy()
correct = results_df[results_df['agent_correct'] == True].copy()

print(f"\nTotal examples:    {len(results_df)}")
print(f"Correct:           {len(correct)} ({len(correct)/len(results_df)*100:.1f}%)")
print(f"Misclassified:     {len(failures)} ({len(failures)/len(results_df)*100:.1f}%)")

# ============================================================
# FAILURE MODE 1: Most confused intent pairs
# ============================================================
print("\n" + "="*50)
print("FAILURE MODE 1: Most Confused Intent Pairs")
print("="*50)
confusion = failures.groupby(['true_intent','agent_intent']).size().reset_index(name='count')
confusion = confusion.sort_values('count', ascending=False)
print(confusion.head(10).to_string(index=False))

# ============================================================
# FAILURE MODE 2: Short/vague messages
# ============================================================
print("\n" + "="*50)
print("FAILURE MODE 2: Short Messages (<80 chars)")
print("="*50)
failures['msg_len'] = failures['message'].str.len()
short_failures = failures[failures['msg_len'] < 80]
print(f"Short message failures: {len(short_failures)}/{len(failures)} ({len(short_failures)/len(failures)*100:.1f}% of failures)")
print("\nExamples:")
for _, row in short_failures.head(5).iterrows():
    print(f"  MSG: '{row['message'][:80]}'")
    print(f"  True: {row['true_intent']} | Predicted: {row['agent_intent']}")
    print()

# ============================================================
# FAILURE MODE 3: Angry/emotional messages
# ============================================================
print("="*50)
print("FAILURE MODE 3: Emotional/Angry Messages")
print("="*50)
angry_words = ['terrible|worst|horrible|unacceptable|furious|angry|'
               'awful|useless|disgusting|pathetic|ridiculous|!!']
angry_mask = failures['message'].str.lower().str.contains(
    angry_words[0], na=False, regex=True
)
angry_failures = failures[angry_mask]
print(f"Angry message failures: {len(angry_failures)}")
print("\nExamples:")
for _, row in angry_failures.head(3).iterrows():
    print(f"  MSG: '{row['message'][:100]}'")
    print(f"  True: {row['true_intent']} | Predicted: {row['agent_intent']}")
    print()

# ============================================================
# FAILURE MODE 4: Multi-intent messages
# ============================================================
print("="*50)
print("FAILURE MODE 4: Likely Multi-Intent (Long messages >200 chars)")
print("="*50)
long_failures = failures[failures['msg_len'] > 200]
print(f"Long message failures: {len(long_failures)}")
print("\nExamples:")
for _, row in long_failures.head(3).iterrows():
    print(f"  MSG: '{row['message'][:200]}'")
    print(f"  True: {row['true_intent']} | Predicted: {row['agent_intent']}")
    print()

# ============================================================
# FAILURE MODE 5: Low quality replies from judge
# ============================================================
print("="*50)
print("FAILURE MODE 5: Low Quality Replies (judge score < 3.5)")
print("="*50)
if 'overall_score' in judge_df.columns:
    bad_replies = judge_df[judge_df['overall_score'] < 3.5]
    print(f"Low quality replies: {len(bad_replies)}/{len(judge_df)}")
    for _, row in bad_replies.head(3).iterrows():
        print(f"\n  Customer: {str(row['customer_message'])[:100]}")
        print(f"  Agent draft: {str(row['agent_draft'])[:100]}")
        print(f"  Score: {row['overall_score']}")
        print(f"  Explanation: {str(row['explanation'])[:150]}")

# ============================================================
# PER-INTENT ACCURACY
# ============================================================
print("\n" + "="*50)
print("PER-INTENT ACCURACY")
print("="*50)
per_intent = results_df.groupby('true_intent')['agent_correct'].agg(['sum','count','mean'])
per_intent.columns = ['correct', 'total', 'accuracy']
per_intent['accuracy'] = per_intent['accuracy'].map('{:.1%}'.format)
print(per_intent.to_string())

# ============================================================
# SAVE REPORT
# ============================================================
report = {
    'summary': {
        'total_evaluated': len(results_df),
        'correct': len(correct),
        'failures': len(failures),
        'accuracy': round(len(correct)/len(results_df), 4),
        'failure_rate': round(len(failures)/len(results_df), 4)
    },
    'failure_modes': {
        'mode1_confused_pairs': confusion.head(5).to_dict('records'),
        'mode2_short_messages': {
            'count': len(short_failures),
            'pct_of_failures': round(len(short_failures)/max(len(failures),1), 3),
            'examples': short_failures['message'].head(3).tolist()
        },
        'mode3_angry_messages': {
            'count': len(angry_failures),
            'examples': angry_failures['message'].head(3).tolist()
        },
        'mode4_long_multi_intent': {
            'count': len(long_failures),
            'examples': long_failures['message'].head(3).tolist()
        },
        'mode5_low_quality_replies': {
            'count': len(bad_replies) if 'overall_score' in judge_df.columns else 0
        }
    },
    'per_intent_accuracy': results_df.groupby('true_intent')['agent_correct'].mean().round(3).to_dict()
}

os.makedirs('results', exist_ok=True)
with open('results/failure_analysis.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Saved failure_analysis.json")
print(f"\n{'='*50}")
print(f"TOP FINDINGS FOR REPORT:")
print(f"{'='*50}")
print(f"1. Overall accuracy: {len(correct)/len(results_df)*100:.1f}%")
print(f"2. Biggest confusion: {confusion.iloc[0]['true_intent']} → {confusion.iloc[0]['agent_intent']} ({confusion.iloc[0]['count']} cases)")
print(f"3. Short msg failures: {len(short_failures)} ({len(short_failures)/max(len(failures),1)*100:.1f}% of failures)")
print(f"4. Angry msg failures: {len(angry_failures)}")
print(f"5. Multi-intent failures: {len(long_failures)}")