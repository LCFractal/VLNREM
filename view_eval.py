import json
import networkx as nx
from r2r_src.utils import load_nav_graphs

from eval import Evaluation

file_path = 'snap/eval_EnvDrop/submit_val_unseen.json'

Evaluation([split], featurized_scans, tok)

self.graphs = load_nav_graphs(self.scans)

for split in splits:
    for item in load_datasets([split]):
        if scans is not None and item['scan'] not in scans:
            continue
        self.gt[str(item['path_id'])] = item
        self.scans.append(item['scan'])
        self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]
