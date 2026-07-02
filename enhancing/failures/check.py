import sys
sys.path.insert(0, '.')
from read_serve import read_serve
import numpy as np

data = read_serve('data/14unL.csv')
markers = [f'marker_{i}' for i in range(14)]
valid = np.ones(len(data['frames']), dtype=bool)
for m in markers:
    valid &= ~np.isnan(data[m]['TX'])

all_valid_count = int(np.sum(valid))
total = len(data['frames'])
indices = np.where(valid)[0]

print(f'All-valid frames: {all_valid_count} out of {total}')
if all_valid_count > 0:
    print(f'First all-valid frame: {indices[0]}')
    print(f'Last all-valid frame:  {indices[-1]}')
else:
    print('No frames with all 14 markers valid')

# Show valid count per marker
print()
print('Valid frames per marker:')
for m in markers:
    n = int(np.sum(~np.isnan(data[m]['TX'])))
    print(f'  {m}: {n}/{total}')
