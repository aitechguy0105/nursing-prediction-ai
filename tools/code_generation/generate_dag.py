"""
cd tools
python code_generation/generate_dag.py --client uch --file-name code_generation/runs --experiment-name "uch-SAIV-751(2)-saiva-3day-hosp-v3" --experiment-date '2021-06-03' --training-start-date '2018-05-01' --s3-folder 156 --incremental-file-prefix UNITEDCHURCHHOMES_V4_DIFF --environment dev
"""

import csv
import pprint
import sys
import fire
from jinja2 import Template, Environment

sys.path.insert(0, '../tools')
from modelid_addition import add_metadata
env = Environment(autoescape=False, optimized=False)

def main(file_name, experiment_name, experiment_date, s3_folder, training_start_date, incremental_file_prefix, client, environment):
    facilities = []
    with open(f'{file_name}.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        first_row = next(csv_reader)
        column_names = first_row.keys()
        start_index = 10 + len(client)
        for col in column_names:
            if col.startswith(f'facility_{client}') and col.endswith('valid_aucroc'):
                facility_name = int(col[start_index:-13])
                facilities.append(facility_name)
        
    with open(f'{file_name}.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        model_dict = {}
        for f in facilities:
            model_dict[f] =  {'model_id': None, 'aucroc': 0.0}
        for row in csv_reader:
            row = dict(row)
            for facility in facilities:
                aucroc = float(row[f'facility_{client}_{facility}_valid_aucroc'])
                if aucroc > model_dict[facility]['aucroc']:
                    model_dict[facility]['aucroc'] = aucroc
                    model_dict[facility]['model_id'] = row['Run ID']
    model_metadata = {}
    for key in model_dict:
        model_dict[key].pop('aucroc')
        model_metadata[model_dict[key]['model_id']] = {
            'model_id': model_dict[key]['model_id'],
            'dayspredictionvalid': 3,
            'predictiontask': 'hospitalization',
            'modeldescription': experiment_name,
            'prospectivedatestart': experiment_date
        }
    
    facility_dict_string = pprint.pformat(model_dict, indent=4)
    with open("code_generation/dag.txt", "r") as f:
        template = env.from_string(f.read())
    
    # Generate dag
    with open(f"../airflow/dags/{client}.py", "w") as f:
        # Stream renders the template to a stream for writing to a file.
        # Pass your variables as kwargs
        template.stream(
            client=client,
            next_ds='{{ next_ds }}',
            incremental_file_prefix=incremental_file_prefix,
            facility_dict_string=facility_dict_string,
        ).dump(f)
    
    # Insert model metadata
    add_metadata(environment, model_metadata)
    
    #Update constants in models
    with open("code_generation/constants.txt", "r") as f:
        template = env.from_string(f.read())
    with open("../models/saiva-3-day-hosp-v3/src/shared/constants.py", "a") as f:
        template.stream(
            client=client,
            s3_folder=s3_folder,
            model_dict=model_dict,
            training_start_date=training_start_date
        ).dump(f)
        
if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(main)