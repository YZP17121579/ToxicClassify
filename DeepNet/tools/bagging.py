import sys
import pandas as pd

"""
Bagging based on Average.
"""

ensembeled = sys.argv[1:]
subs = []
for e in ensembeled:
	print(e)
	subs.append(pd.read_csv(e))

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for sub in subs[1:]:
	for c in classes:
	    subs[0][c] += sub[c]

for c in classes:
	subs[0][c] /= len(subs)

subs[0].to_csv('Bagging.csv', index=False)