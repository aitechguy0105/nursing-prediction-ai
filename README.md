# Saiva AI
#

## Deploying to AWS
1. First - clone this repo down.
2. Create an S3 bucket in the new account (make sure it has no public access) and upload the entire saivahc folder.
3. Create ECR repositories for `webapp`, `etl`, and `performance-dashboard` and build and push the containers into these repos.
4. Create a keypair within the EC2 console and save the `.pem` file locally. This is how we'll access the EC2 servers.
5. Go to CloudFormation within this new account and paste in the link to the `master.yml` file. As a parameter enter in the bucket name you created.
6. CloudFormation will start creating all the resources. Once RDS is created log in and create databases `airflow` and `oscar`, and within `oscar` create all the necessary tables using the `tools/create_table_schema_oscar.sql`. (Credentials will be in SecretsManager).
7. If deploying to a brand new account you'll need to add in a DNS record for AWS Certificate Manager into the `saivahealth.com` DNS record. Navigate to Certificate Manager to see what DNS record to add.
8. For the webapp you'll need to run migrations and create a superuser. See the specific directions in the README in the `webapp` folder.
#
