"""
Copy emails to webapp facilities
Examples:
python copy_emails_from_dag.py --env staging
"""

import json

import boto3
import fire
import psycopg2
import psycopg2.extras
from eliot import log_message

REGION_NAME = "us-east-1"

CLIENTS = {
    'avante': {
        3: {
            'emails': ["wmilam@avantegroup.com", "JHand@avantecenters.com", "SJoubert@avantecenters.com",
                       "Rruohonen@avantecenters.com", "KRatanasurakarn@avantegroup.com"],
            'model_id': '270cc0c6b3d240c88ac15ecab24e6790',
            'valid_email_domains': ['avante']
        },
        4: {
            'emails': ["wmilam@avantegroup.com", "psurtain@avantecenters.com", "Asfoster@avantecenters.com",
                       "Lerobinson@avantecenters.com", "Ejeanpierre@avantecenters.com"],
            'model_id': '270cc0c6b3d240c88ac15ecab24e6790',
            'valid_email_domains': ['avante']
        },
        5: {
            'emails': ["kervin@avantecenters.com", "sbruno@avantegroup.com", "wmilam@avantecenters.com",
                       "tbenjamin@avantecenters.com"],
            'model_id': '49d6cc5035354a958f405db8cd7f8beb',
            'valid_email_domains': ['avante']
        },
        6: {
            'emails': ["wmilam@avantegroup.com", "awoodman@avantecenters.com", "bicrawford@avantecenters.com",
                       "BNunn@avantecenters.com", "sbruno@avantegroup.com"],
            'model_id': '647740c9d0df4de8963f8cf8ce03f909',
            'valid_email_domains': ['avante']
        },
        7: {
            'emails': ["clawrenceRice@avantegroup.com",
                       "wmilam@avantegroup.com",
                       "APierce@avantecenters.com",
                       "SRabello@avantecenters.com"],
            'model_id': '270cc0c6b3d240c88ac15ecab24e6790',
            'valid_email_domains': ['avante']
        },
        8: {
            'emails': ["wmilam@avantegroup.com", "GStrunk-Gamel@avantecenters.com", "mmcdowell@avantecenters.com",
                       "KRatanasurakarn@avantegroup.com"],
            'model_id': '270cc0c6b3d240c88ac15ecab24e6790',
            'valid_email_domains': ['avante']
        },
        9: {
            'emails': ["rbachman@avantecenters.com", "kratanasurakarn@avantegroup.com", "wmilam@avantecenters.com",
                       "maugustin@avantecenters.com"],
            'model_id': '647740c9d0df4de8963f8cf8ce03f909',
            'valid_email_domains': ['avante']
        },
        10: {
            'emails': ["wmilam@avantegroup.com", "sbruno@avantegroup.com", "kApodaca@avantecenters.com",
                       "MVelezMaldonado@avantecenters.com", "nsierra@avantecenters.com"],
            'model_id': '270cc0c6b3d240c88ac15ecab24e6790',
            'valid_email_domains': ['avante']
        },
        13: {
            'emails': ["AMuniz@avantecenters.com", "wmilam@avantegroup.com", "clawrenceRice@avantegroup.com",
                       "ljames@avantecenters.com", "cescander@avantecenters.com"],
            'model_id': '067b8765afaf4f769c8388532eada783',
            'valid_email_domains': ['avante']
        },
        21: {
            'emails': ["wmilam@avantegroup.com", "mgeorge@avantecenters.com", "Afortner@avantecenters.com",
                       "sbruno@avantegroup.com", "nkypriotes@avantecenters.com"],
            'model_id': '71f1c512d7ee4c18994f5426dda67172',
            'valid_email_domains': ['avante']
        },
        1: {
            'emails': ["wmilam@avantegroup.com", "utheoc@avantecenters.com", "SHenriques@avantecenters.com",
                       "cparramore@avantecenters.com"],
            'model_id': '71f1c512d7ee4c18994f5426dda67172',
            'valid_email_domains': ['avante']
        },
    },

    'dycora': {
        121: {
            'emails': ['Teresa.Mendoza@dycora.com',
                       'Dionicia.Calvan@dycora.com',
                       'Pepito.Quezon@dycora.com',
                       'MaryLou.Jenkins@dycora.com',
                       'Mary-beth.newell@dycora.com',
                       'Shelley.wright@dycora.com',
                       'Tara.raymond@dycora.com'],
            'model_id': '41238147855f4210ae98e067241f8dff'
        },
        302: {
            'emails': ['Mary-beth.newell@dycora.com',
                       'Shelley.wright@dycora.com',
                       'Tara.raymond@dycora.com',
                       'Ken.Evans@dycora.com',
                       'Esmeralda.Palma@dycora.com',
                       'Anna.Canedo@dycora.com',
                       'Wen.Wu@dycora.com',
                       'Donna.Rogers@dycora.com'],
            'model_id': 'e6b3ffdf095747a7891746b01193db66'
        },
    },

    'infinity-benchmark': {
        75: {
            'emails': ['jaikumar@saivahc.com'],
            'model_id': '8b099bee6027412da95811bf01af3f80'
        },
        89: {
            'emails': ['jaikumar@saivahc.com'],
            'model_id': '8b099bee6027412da95811bf01af3f80'
        },
    },

    'gulfshore': {
        16: {
            'emails': ['Angela.eckard@gulfshorecc.com', 'louise.merrick@gulfshorecc.com',
                       'Frantz.David@gulfshorecc.com'],
            'model_id': 'fa9a603ea547452799d952475f40b80d'
        },
    },

    'infinity-infinity': {
        75: {
            'emails': ['twebster@infinityoftn.com',
                       'administrator@watersofrobertson.com',
                       'don@watersofrobertson.com',
                       'kmckee@infinityofar.com',
                       'tarmstrong@infinityoffl.com'],
            'model_id': '9bfb02fa32f644aeafba299380742580',
            'valid_email_domains': ['infinity',
                                    'watersofrobertson']
        },
        89: {
            'emails': ['kburnett@saivahc.com'],
            'model_id': 'd32d239731844f4497e27edc51242b62',
            'valid_email_domains': ['infinity',
                                    'watersofrobertson']
        },
    },

    'northshore': {
        10: {
            'emails': ['madhur@saivahc.com'],
            'model_id': '68ca9a93a6c24567a5570a687fb16033'
        },
        13: {
            'emails': ['madhur@saivahc.com'],
            'model_id': 'f27dd0a1791b4c7aa9a7422547620fc5'
        },
    },

    'palmgarden': {
        7: {
            'emails': ['stacey.lopriore@palmgarden.com'],
            'model_id': '4f09a40ceccb4a7e9e25b909b8d51856'
        },
        13: {
            'emails': ['stacey.lopriore@palmgarden.com'],
            'model_id': '4f09a40ceccb4a7e9e25b909b8d51856'
        },
    },

    'trio': {
        7: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'MGreen@trio-healthcare.com',
                       'asimpson@trio-healthcare.com',
                       'Astrickland@fredericksburgrehab.com',
                       'Mkreck@fredericksburgrehab.com'],
            'model_id': '73861fd9a0a5485cb3deccf816a15c7b',
            'valid_email_domains': ['trio',
                                    'fredericksburg']
        },
        42: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'MGreen@trio-healthcare.com',
                       'asimpson@trio-healthcare.com',
                       'aparks@alleghanyrehab.com',
                       'jwilliams@alleghanyrehab.com'],
            'model_id': '31edea3de43f4721bb925c5f146a3189',
            'valid_email_domains': ['trio',
                                    'alleghany']
        },
        52: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'MGreen@trio-healthcare.com',
                       'asimpson@trio-healthcare.com',
                       'mgreeley@martinsvillerehab.com',
                       'sfrazier@martinsvillerehab.com',
                       'ajoyce@martinsvillerehab.com'],
            'model_id': 'e9f2b07d26984dcfbdaff0a86f033e36',
            'valid_email_domains': ['trio',
                                    'martinsville']
        },
        55: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'MGreen@trio-healthcare.com',
                       'asimpson@trio-healthcare.com ',
                       'FGilchrist@portsmouthrehab.com',
                       'LPitts@portsmouthrehab.com',
                       'Zlee@portsmouthrehab.com'],
            'model_id': 'daac8b0f079d487d96046e9dff6efe84',
            'valid_email_domains': ['trio',
                                    'portsmouth']
        },

        265: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'ttaylor@rehabatbayside.com',
                       'Sfallin@rehabatbayside.com'],
            'model_id': 'daac8b0f079d487d96046e9dff6efe84',
            'valid_email_domains': ['trio',
                                    'bayside']
        },
        186: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'squick@eacrumprehab.com'],
            'model_id': 'c77d3159cc044c14bf15da77eb889a17',
            'valid_email_domains': ['trio',
                                    'eacrump']
        },
        21: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'wviers@galaxrehab.com',
                       'teichner@galaxrehab.com'],
            'model_id': 'c77d3159cc044c14bf15da77eb889a17',
            'valid_email_domains': ['trio',
                                    'galax']
        },
        1: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'cmartin@rosehillrehabcenter.com',
                       'rfrye@rosehillrehabcenter.com'],
            'model_id': 'daac8b0f079d487d96046e9dff6efe84',
            'valid_email_domains': ['trio',
                                    'rosehill']
        },
        194: {
            'emails': ['ajohnson@trio-healthcare.com',
                       'jfelts@trio-healthcare.com',
                       'Dworkman@trio-healthcare.com',
                       'jcoleman@shenandoahrehabcenter.com',
                       'ssmith@shenandoahrehabcenter.com'],
            'model_id': 'c77d3159cc044c14bf15da77eb889a17',
            'valid_email_domains': ['trio',
                                    'shenandoah']
        },
        273: {
            'emails': ['kclaytor@trio-healthcare.com',
                       'rengleka@trio-healthcare.com',
                       'SRose@trio-healthcare.com',
                       'tcarr@beavercreekrehab.com',
                       'tclevenger@beavercreekrehab.com'
                       ],
            'model_id': 'daac8b0f079d487d96046e9dff6efe84',
            'valid_email_domains': ['trio',
                                    'beavercreek']
        },
        274: {
            'emails': ['kclaytor@trio-healthcare.com',
                       'rengleka@trio-healthcare.com',
                       'SRose@trio-healthcare.com',
                       'krule@bellbrookrehab.com',
                       'aivey@bellbrookrehab.com'
                       ],
            'model_id': 'c77d3159cc044c14bf15da77eb889a17',
            'valid_email_domains': ['trio',
                                    'bellbrook']
        },
        275: {
            'emails': ['kclaytor@trio-healthcare.com',
                       'rengleka@trio-healthcare.com',
                       'SRose@trio-healthcare.com',
                       'wramsey@centervillerehab.com',
                       'bcarlson@centervillerehab.com'
                       ],
            'model_id': 'c77d3159cc044c14bf15da77eb889a17',
            'valid_email_domains': ['trio',
                                    'centerville']
        },
        276: {
            'emails': ['kclaytor@trio-healthcare.com',
                       'rengleka@trio-healthcare.com',
                       'SRose@trio-healthcare.com',
                       'jlicata@rehabatenglewood.com',
                       'norock@rehabatenglewood.com'
                       ],
            'model_id': '31edea3de43f4721bb925c5f146a3189',
            'valid_email_domains': ['trio',
                                    'englewood']
        },
        277: {
            'emails': ['kclaytor@trio-healthcare.com',
                       'rengleka@trio-healthcare.com',
                       'SRose@trio-healthcare.com',
                       'jdeards@jamestownplacerehab.com',
                       'mlavigne@xeniarehab.com'
                       ],
            'model_id': 'e9f2b07d26984dcfbdaff0a86f033e36',
            'valid_email_domains': ['trio',
                                    'jamestownplace',
                                    'xenia']
        },
        278: {
            'emails': ['kclaytor@trio-healthcare.com',
                       'rengleka@trio-healthcare.com',
                       'SRose@trio-healthcare.com',
                       'lhanley@rehabatportsmouth.com',
                       'jelliot@rehabatportsmouth.com'
                       ],
            'model_id': 'c77d3159cc044c14bf15da77eb889a17',
            'valid_email_domains': ['trio',
                                    'portsmouth']
        },
        279: {
            'emails': ['kclaytor@trio-healthcare.com',
                       'rengleka@trio-healthcare.com',
                       'SRose@trio-healthcare.com',
                       'mlavigne@xeniarehab.com',
                       'kcrossley@xeniarehab.com'
                       ],
            'model_id': '31edea3de43f4721bb925c5f146a3189',
            'valid_email_domains': ['trio',
                                    'xenia']
        }
    }
}


class DbEngine(object):
    """
    """

    def __init__(self, region_name=REGION_NAME):
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

    def get_postgres_engine(self, env, db_name):
        """
        Based on the environment connects to the respective database
        :param client: client name
        :return: Saivadb Postgres engine
        """
        log_message(message_type='info', action_type='connect_to_postgresdb', client='SaivaDB')
        # Fetch credentials from AWS Secrets Manager
        postgresdb_info = self.get_secrets(secret_name=f'{env}-saivadb')

        postgresdb_info['dbname'] = db_name

        # Create db connection
        connection = psycopg2.connect(
            user=postgresdb_info['username'],
            password=postgresdb_info['password'],
            host=postgresdb_info['host'],
            port=postgresdb_info['port'],
            database=postgresdb_info['dbname']
        )
        # Return Postgres Engine
        return connection


def run(env='dev', db_name='webapp'):
    db = DbEngine()
    connection = db.get_postgres_engine(env=env, db_name=db_name)

    for client, obj in CLIENTS.items():
        query1 = f"select wr.id from webapp_region wr \
                    JOIN webapp_organization wo ON (wo.id = wr.organization_id) \
                    where wo.org_id ='{client}' and wr.name='DEFAULT_REGION' "

        with connection.cursor() as cursor:
            cursor.execute(query1)
            region_id = cursor.fetchone()[0]

            for facility_id, data in obj.items():
                emails = ','.join(data['emails'])
                if 'valid_email_domains' in data:
                    valid_email_domains = ','.join(data['valid_email_domains'])
                else:
                    valid_email_domains = ''

                query2 = f"update webapp_facility set report_subscribers='{emails}', subscribers_domain='{valid_email_domains}' \
                            where facility_id='{facility_id}' and region_id={region_id}"

                print(query2)
                print('**********************************************')
                cursor.execute(query2)
                connection.commit()


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(run)
