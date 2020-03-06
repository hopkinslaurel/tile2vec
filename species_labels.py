import csv

def get_records():
    records = {}
    with open('OR_2011_synthetic_responses.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            records[row['img_name']] = {'forest':int(row['response_forest']), 'forest_no_frag':int(row['response_forest_no_frag']), \
                'forest_no_frag_1':int(row['response_forest_no_frag']), 'desert':int(row['response_desert']), \
                'desert_no_frag':int(row['response_desert_no_frag']), 'frag':int(row['response_frag']), \
                'interaction':int(row['response_interaction'])}
            
    return records
