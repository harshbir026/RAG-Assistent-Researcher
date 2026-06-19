import json
from collections import Counter

with open('golden_qa.json') as f:
    data = json.load(f)

domains = Counter(q['domain'] for q in data['questions'])
types = Counter(q['question_type'] for q in data['questions'])

print("Domain breakdown:")
for d, c in domains.items():
    print(f"  {d}: {c}")

print("\nQuestion type breakdown:")
for t, c in types.items():
    print(f"  {t}: {c}")
