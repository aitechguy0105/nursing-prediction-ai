from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config
from eliot import log_message

AWS_REGION_NAME = "us-east-1"

def get_bedrock_client(region: Optional[str] = None,):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    if region is None:
        target_region = AWS_REGION_NAME
    else:
        target_region = region

    log_message(
                message_type='info',
                result=f"Create new client: \n  Using region: {target_region}"
            )
    session_kwargs = {"region_name": target_region}

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)
    bedrock_client = session.client(
        service_name="bedrock-runtime",
        config=retry_config,
        region_name=AWS_REGION_NAME,
    )

    log_message(
                message_type='info',
                result="boto3 Bedrock client successfully created!"
            )
    return bedrock_client, session
