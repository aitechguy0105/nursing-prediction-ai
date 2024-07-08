import json
import os

from s3fs import S3FileSystem

S3FS = S3FileSystem()
ENV = os.environ.get('SAIVA_ENV', 'dev')


def fetch_report_config(*, report_version):
    """show_progress_notes: key works as a switch to on/off progress_notes.
    run_rule_engine: key helps in enabling and disabling RuleEngine
    """

    try:
        config_path = f's3://saiva-{ENV}-data-bucket/config/report/{report_version}/report_config.json'
        # TODO: update for multimodel - we need to move data (in future maybe create separate files for all clients)
        with S3FS.open(config_path, 'rb') as f:
            return json.load(f)
    except Exception:
        return {}


def get_config_value(config_dict, client, keys):
    """ Check client specific config, if not check
     default config.
    """
    return_list = []
    for key in keys:
        if key in config_dict.get(client, {}):
            return_list.append(config_dict[client][key])
        elif key in config_dict['default']:
            return_list.append(config_dict['default'][key])
        else:
            raise Exception(f'{key} Not found in {config_dict}')
    return return_list
