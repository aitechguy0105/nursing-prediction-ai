# Usage: python .circleci/process_code_scan.py <code_scan_results_path> <slack_token> <slack_channel> <git_commit_sha> <commit_user> <repo_name>

import json
import os
import sys
from dataclasses import dataclass, asdict
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

CODE_SCAN_REULTS_PATH = sys.argv[1]
SLACK_TOKEN = sys.argv[2]
SLACK_CHANNEL = sys.argv[3]
GIT_COMMIT_SHA = sys.argv[4]
COMMIT_USER = sys.argv[5]
REPO_NAME = sys.argv[6]

@dataclass
class Issue:
    title: str
    description: str
    severity: str
    file_name: str
    file_path: str
    line_number: int


def process_findings() -> list[Issue]:
    # Read code scan results json path from argumnts
    sanitized_path = os.path.relpath(os.path.join("/", CODE_SCAN_REULTS_PATH), "/")

    # Load JSON file from path
    with open(sanitized_path, "r", encoding="UTF-8") as f:
        code_scan_results = json.load(f)

    findings = []

    # Process findings
    for finding in code_scan_results["findings"]:
        issue = Issue(
            title=finding["title"],
            description=finding["description"],
            severity=finding["severity"],
            file_name=finding["vulnerability"]["filePath"]["name"],
            file_path=finding["vulnerability"]["filePath"]["path"],
            line_number=finding["vulnerability"]["filePath"]["startLine"],
        )
        findings.append(issue)
        print("Found issue: {}".format(asdict(issue)))

    return findings


def post_findings_to_slack(*, findings: list[Issue]) -> None:
    """
    Post findings to slack
    """
    print("Posting findings to slack")

    client = WebClient(token=SLACK_TOKEN)

    critical_issues = 0
    for finding in findings:
        if finding.severity.lower() == "critical":
            critical_issues += 1

    branch_name = CODE_SCAN_REULTS_PATH.rstrip(".json")

    codeguru_url = "https://us-east-1.console.aws.amazon.com/codeguru/security/findings"
    codeguru_filters = f'?region=us-east-1#filter={{"tokens":[{{"propertyKey":"scanName","operator":"=","value":"{branch_name}"}}],"operation":"and"}}'

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Code scan found {len(findings)} issues",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Repository*\n{REPO_NAME}"},
                {"type": "mrkdwn", "text": f"*Commit owner*\n{COMMIT_USER}"},
                {"type": "mrkdwn", "text": f"*Branch*\n{branch_name.lstrip(f'{REPO_NAME}-')}"},
                {"type": "mrkdwn", "text": f"*Commit SHA*\n{GIT_COMMIT_SHA}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Issues*\n{len(findings)}"},
                {"type": "mrkdwn", "text": f"*Critical*\n{critical_issues}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "emoji": True,
                        "text": "View issues",
                    },
                    "style": "primary",
                    "value": "View issues",
                    "url": codeguru_url + codeguru_filters,
                }
            ],
        },
    ]

    try:
        client.chat_postMessage(channel=SLACK_CHANNEL, blocks=blocks)
    except SlackApiError as e:
        print(f"Got an error: {e.response['error']}")
        sys.exit(1)

    if critical_issues > 0:
        sys.exit(0)


if __name__ == "__main__":
    # Process findings
    findings = process_findings()

    # Fail if there are issues and post to slack
    if findings:
        print(f"Code scan found {len(findings)} issues")
        post_findings_to_slack(findings=findings)

    else:
        print("Code scan found no issues")
