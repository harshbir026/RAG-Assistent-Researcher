import json
from collections import Counter

with open('golden_qa.json') as f:
    data = json.load(f)

tiers = Counter(q['tier_name'] for q in data['questions'])
for t, c in sorted(tiers.items()):
    print(f"  {t}: {c}")
