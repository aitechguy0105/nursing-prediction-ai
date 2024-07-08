import json
import os
import boto3
from locust import HttpUser, task, between


ENV = os.environ.get("SAIVA_ENV", "dev")
region_name = "us-east-1"
session = boto3.session.Session()
secrets_client = session.client(
    service_name="secretsmanager", region_name=region_name
)

class QuickstartUser(HttpUser):
    wait_time = between(1, 5)

    def headers(self):
        return {"Authorization": f"Bearer {self.access_token}", 'accept': 'application/json'}

    def on_start(self):
        self.demo_user_info = json.loads(secrets_client.get_secret_value(
            SecretId=f'{ENV}-demo-user')['SecretString']
        )
        resp = self.client.post("token/", json={"username":self.demo_user_info['username'], "password":self.demo_user_info['otp']})
        self.access_token = resp.json()['access']
        self.refresh_token = resp.json()['refresh']
        self.org = 'saiva_demo'
        self.facility_id = 265
        self.report_id = 'saiva_demo_265_2022-05-15_scheduled__2022-05-14T07:30:00+00:00'
    
    @task
    def supported_app_versions(self):
        self.client.get('supported-app-versions/ios/')

    @task
    def user_details(self):
        self.client.get(
            f"users/{self.demo_user_info['username']}/",
            headers=self.headers(),
        )

    @task
    def report(self):
        self.client.get(
            f'report/{self.org}/{self.facility_id}/',
            headers=self.headers(),
        )

    @task
    def visits(self):
        self.client.get(
            f'visits/{self.report_id}',
            headers=self.headers(),
        )
