import base64
from typing import List
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import boto3
import os
from pydantic import BaseModel

app = FastAPI()


class ConverseRequest(BaseModel):
    messages: List[dict] = []
    modelId: str
    region: str
    system: List[dict] | None = None


@app.post("/api/converse")
async def converse(request: ConverseRequest):
    model_id = request.modelId
    region = request.region
    if region == '':
        region = request.region

    try:
        client = boto3.client("bedrock-runtime",
                              region_name=region)
        max_tokens = 200
        for message in request.messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if 'image' in content:
                        image_bytes = base64.b64decode(content['image']['source']['bytes'])
                        content['image']['source']['bytes'] = image_bytes
                    if 'video' in content:
                        video_bytes = base64.b64decode(content['video']['source']['bytes'])
                        content['video']['source']['bytes'] = video_bytes
                    if 'document' in content:
                        document_bytes = base64.b64decode(content['document']['source']['bytes'])
                        content['document']['source']['bytes'] = document_bytes
        command = {
            "inferenceConfig": {"maxTokens": max_tokens},
            "messages": request.messages,
            "modelId": model_id
        }
        if request.system is not None:
            command["system"] = request.system

        def event_generator():
            try:
                response = client.converse_stream(**command)
                complete_res = ''
                for item in response['stream']:
                    if "contentBlockDelta" in item:
                        text = item["contentBlockDelta"].get("delta", {}).get("text", "")
                        if text:
                            complete_res += text
                print(complete_res)
                return complete_res
            except Exception as err:
                return f"Error: {str(err)}"
        res = event_generator()
        return {"result": res}

    except Exception as error:
        return PlainTextResponse(f"Error: {str(error)}", status_code=500)


if __name__ == "__main__":
    print("Starting webserver...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
