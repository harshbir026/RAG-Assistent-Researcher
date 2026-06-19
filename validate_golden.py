import json

with open('golden_qa.json') as f:
    data = json.load(f)

print(f"Total questions: {len(data['questions'])}")
for i, q in enumerate(data['questions'], 1):
    assert q['ground_truth_answer'], f"Q{i} missing answer: {q['question']}"
    assert q['relevant_paper_ids'], f"Q{i} missing paper IDs: {q['question']}"
print("✅ All questions have answers and paper IDs filled in.")
