import boto3
import json
from botocore.exceptions import ClientError
import os 

def invoke_agent(agent_id, agent_alias_id, session_id, prompt):
    try:
        client = boto3.session.Session().client(service_name="bedrock-agent-runtime")
        # See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            enableTrace=True,
            sessionId=session_id,
            inputText=prompt,
        )

        output_text = ""
        citations = []
        trace = {}

        for event in response.get("completion"):
            # Combine the chunks to get the output text
            if "chunk" in event:
                chunk = event["chunk"]
                output_text += chunk["bytes"].decode()
                if "attribution" in chunk:
                    citations = citations + chunk["attribution"]["citations"]

            # Extract trace information from all events
            if "trace" in event:
                for trace_type in ["preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                    if trace_type in event["trace"]["trace"]:
                        if trace_type not in trace:
                            trace[trace_type] = []
                        trace[trace_type].append(event["trace"]["trace"][trace_type])

    except ClientError as e:
        raise

    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }


def invoke_model(encoded_image):
    try:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded_image
                            }
                        },
                        {
                            "type": "text",
                            "text": "Now you are going to be a image description generator. With the input image. You are going to give a detailed description of the image. You have to look for product, packaging in the image, any damages in those. You have to specify the place of damage if any. You have to say the color and describe the product shortly too."
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "anthropic_version": "bedrock-2023-05-31"
        }

        modelclient = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION")
        )

        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

        modelresponse = modelclient.invoke_model(
            modelId=model_id,
            contentType="application/json",
            body=json.dumps(payload)
        )
        output_binary = modelresponse["body"].read()
        output_json = json.loads(output_binary)
        output = output_json["content"][0]["text"]

        print(output)

    except ClientError as e:
        raise
    
    return output