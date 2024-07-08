"""
cd tools
python code_generation/generate_client.py --client uch --include_labs True \
  --emar-types "['eMar - Hour of Administration Level Note','eMar - Shift Level Administration Note']" \
  --training-start-date '2018-05-01' \
  --training-end-date '2021-04-30' \
  --census-action-codes "('DAMA', 'DD', 'DE', 'DH', 'HP', 'HUP', 'L', 'LOA', 'LV', 'MHHP', 'MHHU', 'MHTP', 'MHTU', 'MO', 'RDD', 'RDE', 'RDH', 'TO', 'TP', 'TUP')"
"""

import fire
from jinja2 import Template, Environment

env = Environment(autoescape=False, optimized=False)

def main(client, emar_types, training_start_date, training_end_date, census_action_codes, include_labs=False):
    with open("code_generation/client.txt", "r") as f:
        template = env.from_string(f.read())

    with open(f"../models/saiva-3-day-hosp-v3/src/clients/{client}.py", "w") as f:
      
        template.stream(
          class_name=client.capitalize(),
          include_labs=include_labs,
          emar_types=emar_types,
          training_start_date=training_start_date,
          training_end_date=training_end_date,
          census_action_codes=census_action_codes
        ).dump(f)

if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(main)