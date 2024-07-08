import csv
import datetime
import os

def save_trials(study, trial):
    with open('trial_data.csv', 'a', newline='') as csvfile:
        fieldnames = [
            'trial_number', 
            'params', 
            'value', 
            'datetime_start', 
            'datetime_complete',
            'step_name',
            'lgbm_params',
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if csvfile.tell() == 0:
            writer.writeheader()
        
        step_name = trial.system_attrs.get('lightgbm_tuner:step_name', '')
        lgbm_params = trial.system_attrs.get('lightgbm_tuner:lgbm_params', '')
        
        writer.writerow({
            'trial_number': trial.number,
            'params': str(trial.params),
            'value': trial.value,
            'datetime_start': trial.datetime_start.strftime('%Y-%m-%d %H:%M:%S'),
            'datetime_complete': trial.datetime_complete.strftime('%Y-%m-%d %H:%M:%S'),
            'step_name': step_name,
            'lgbm_params': lgbm_params,
        })
