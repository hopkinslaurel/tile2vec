import csv
import numpy as np

def import_records():
    records = {}
    with open('OR_2011_synthetic_responses.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            records[row['img_name']] = {'forest':int(row['response_forest']), 'forest_no_frag':int(row['response_forest_no_frag']), \
                    'forest_no_frag_1':int(row['response_forest_no_frag']), 'forest_no_frag_tca':int(row['response_forest_no_frag_tca']), \
                'desert':int(row['response_desert']), 'desert_no_frag':int(row['response_desert_no_frag']), 'frag':int(row['response_frag']), \
                'frag_tca':int(row['response_frag_tca']), 'interaction':int(row['response_interaction']), 'interaction_tca':int(row['response_interaction_tca'])}
    return records

def get_records(triplet_idx, species):
    records = import_records()
    y = np.zeros(len(triplet_idx))
    idx = 0
    for triplet in triplet_idx:
        y[idx] = records[str(triplet)][species]
        idx += 1

    return y
