import json

with open('golden_qa.json') as f:
    data = json.load(f)

questions = data['questions']

# Map question_type -> tier (with manual overrides for borderline cases)
TIER_3_OVERRIDE_QUESTIONS = [
    "How does deepfake detection research connect to broader synthetic media and misinformation concerns?",
    "How does secure multi-party computation compare to differential privacy for privacy-preserving machine learning?",
    "How do federated learning approaches differ when applied to healthcare versus general IoT settings?",
]

tier1_count, tier2_count, tier3_count = 0, 0, 0

for i, q in enumerate(questions):
    qid_prefix = None

    if q['question'] in TIER_3_OVERRIDE_QUESTIONS or q['question_type'] == 'multi_hop':
        q['tier'] = 3
        q['tier_name'] = 'multi_hop'
        q['expected_retriever'] = 'agent'
        tier3_count += 1
        qid_prefix = f"t3_{tier3_count:03d}"
    elif q['question_type'] in ('comparison', 'summarisation'):
        q['tier'] = 2
        q['tier_name'] = 'synthesis'
        q['expected_retriever'] = 'hybrid'
        tier2_count += 1
        qid_prefix = f"t2_{tier2_count:03d}"
    else:  # factual
        q['tier'] = 1
        q['tier_name'] = 'factual'
        q['expected_retriever'] = 'dense'
        tier1_count += 1
        qid_prefix = f"t1_{tier1_count:03d}"

    q['id'] = qid_prefix
    # rename ground_truth_answer -> ground_truth to match patch spec
    q['ground_truth'] = q.pop('ground_truth_answer')

with open('golden_qa.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Tier 1 (factual)  : {tier1_count}")
print(f"Tier 2 (synthesis): {tier2_count}")
print(f"Tier 3 (multi_hop): {tier3_count}")
print(f"Total: {tier1_count + tier2_count + tier3_count}")
