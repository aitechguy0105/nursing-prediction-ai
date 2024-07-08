import boto3
from eliot import log_message
import json
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import fire

# fillin the correct model metadata before running the code.
model_metadata = {
    'facilityid-type1': {
        'model_id': '418e412613f84b308ef88c522decbcbc',
        'dayspredictionvalid': 3,
        'predictiontask': 'hospitalization',
        'modeldescription': 'trio_retrained_onemonth_valid',
        'prospectivedatestart': '2020-10-02'

    },
    'facilityid-type2': {
        'model_id': '44f99f4cebf64128bd430382cf7c0a14',
        'dayspredictionvalid': 3,
        'predictiontask': 'hospitalization',
        'modeldescription': 'trio_retrained_onemonth_valid',
        'prospectivedatestart':'2020-10-02'

    },
    'facilityid-type3': {
        'model_id': '5b21bc80dec24ed28ecd54f5219139c6',
        'dayspredictionvalid': 3,
        'predictiontask': 'hospitalization',
        'modeldescription': 'trio_retrained_onemonth_valid',
        'prospectivedatestart':'2020-10-02'

    },
    'facilityid-type4': {
        'model_id': '754618bc685547568f7d12bf7e11c6fd',
        'dayspredictionvalid': 3,
        'predictiontask': 'hospitalization',
        'modeldescription': 'trio_retrained_onemonth_valid',
        'prospectivedatestart':'2020-10-02'

    },
    'facilityid-type5': {
        'model_id': '17898acf4ba74d1698b3acac6cb29992',
        'dayspredictionvalid': 3,
        'predictiontask': 'hospitalization',
        'modeldescription': 'trio_retrained_onemonth_valid',
        'prospectivedatestart': '2020-10-02'

    },
    'facilityid-type6': {
        'model_id': 'b9fb5010ecc2421ab5e3e7fdf8835e0a',
        'dayspredictionvalid': 3,
        'predictiontask': 'hospitalization',
        'modeldescription': 'trio_retrained_onemonth_valid',
        'prospectivedatestart':'2020-10-02'

    },
    'facilityid-type7': {
        'model_id': 'd3c0d3d335ec483da652665221aabf04',
        'dayspredictionvalid': 3,
        'predictiontask': 'hospitalization',
        'modeldescription': 'trio_retrained_onemonth_valid',
        'prospectivedatestart':'2020-10-02'

    },
    'facilityid-type8': {
        'model_id': 'a06f7d2408324abdac6cbca9fc1b7e7d',
        'dayspredictionvalid': 3,
        'predictiontask': 'hospitalization',
        'modeldescription': 'trio_retrained_onemonth_valid',
        'prospectivedatestart':'2020-10-02'

    }

}


class DbEngine():
    """
    Fetch the credentials from AWS Secrets Manager.
    :return: DB connection to the respective database
    """

    def __init__(self, region_name='us-east-1'):
        self.session = boto3.session.Session()
        self.secrets_client = self.session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

    def get_secrets(self, secret_name):
        """
        :return: Based on the environment get secrets for
        Client SQL db & Postgres Saivadb
        """
        log_message(message_type='info', action_type='get_secrets', secret_name=secret_name)
        db_info = json.loads(
            self.secrets_client.get_secret_value(SecretId=secret_name)[
                'SecretString'
            ]
        )
        return db_info

    def get_postgresdb_engine(self, env):
        """
        Based on the environment connects to the respective database
        :return: Saivadb Postgres engine
        """
        log_message(message_type='info', action_type='connect_to_postgresdb', client='SaivaDB')
        # Fetch credentials from AWS Secrets Manager
        postgresdb_info = self.get_secrets(secret_name=f'{env}-saivadb')
        log_message(
            message_type="info",
            result=f"Connection with postgres-{env} db established.",
        )
        # Create DB URL
        saivadb_url = URL(
            drivername='postgresql',
            username=postgresdb_info['username'],
            password=postgresdb_info['password'],
            host=postgresdb_info['host'],
            port=postgresdb_info['port'],
            database=postgresdb_info['dbname'],
        )
        # Return Postgres Engine
        return create_engine(saivadb_url, echo=False)


def add_metadata(env, metadata=None):
    """
    1. connect with postgres saiva db.
    2. convert the dict into list of tuples form.
    3. insert the tuple values into the saiva db.
    
    :param env:
    :return:
    """
    if metadata is None:
        metadata = model_metadata
    postgresdb = DbEngine()
    args = []
    for key, value in metadata.items():
        metadata[key]['modeldescription'] += '_' + key
        temp = list(value.values())
        args.append(tuple(temp))
    if env == 'all':
        ENV = ['dev', 'staging', 'prod']
    else:
        ENV = [env]
    for env in ENV:
        print('@' * 15, f'execution running for {env}', '@' * 15)
        saivaengine = postgresdb.get_postgresdb_engine(env)
        for arg_str in args:
            saivaengine.execute(
                f"""insert into saivadb.public.model_metadata
                (modelid, dayspredictionvalid, predictiontask, modeldescription, prospectivedatestart)
                VALUES {arg_str};
                """)
        print('@' * 15, f'execution completed for {env}', '@' * 15)
    print('@' * 15, 'execution completed, exiting now !!! ', '@' * 15)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(add_metadata)

# python modelid_addition.py --env all
# python modelid_addition.py --env dev
