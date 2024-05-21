# LineBot+gpt-3.5_fine tuning

Owner: Tang Hsiang
Tags: AI
Date: December 27, 2023 9:55 AM
Status: Done

**Process Directory Structure Diagram**

```
LineBot+gpt-3.5_fine tuning/
├── Reference to OpenAI GPT-3.5 Official Documentation
├── Collecting Training Data
│   ├── Introducing Claude
│   └── Preparing the Training Dataset
├── Creating the Dataset
│   ├── Visiting OpenAI Fine-tuning Page
│   └── Preparing Data in JSON Format
├── Writing Fine Tuning Code
│   ├── Setting Up the Programming Environment
│   ├── Code Writing
│   └── Reference to YouTube Tutorial Videos
├── Developing LineBot Application
│   ├── Writing and Adjusting Code
│   └── Problems Encountered and Solutions
├── Deploying the Application
│   ├── Deploying LineBot on Render
│   └── Cronjob Setup
├── Comparison Before and After Fine-Tuning
└── Debugging Approach
    ├── Debugging for Specific Issues
    └── Solutions and Thought Process

```

## **1. Collecting Training Data**[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#1-collecting-training-data)

### Generating Q&A Set[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#generating-qa-set)

Introducing an artificial intelligence named Claude, developed by Anthropic. Unlike ChatGPT 3.5, Claude possesses a file upload feature, enabling it to extract information from PDFs or generate summaries. Currently, Claude is available only in Europe and America. To use Claude, users need to change their IP to the Americas via the Opera browser and complete registration using an SMS service. Once registered, users can query Claude just like they would with ChatGPT. (Available to users in Taiwan from 2023/10/18, no need to switch VPN)

![https://yushaing.vercel.app/assets/images/1-179aa891e12d13b79d52a9bd05801a97.png](https://yushaing.vercel.app/assets/images/1-179aa891e12d13b79d52a9bd05801a97.png)

Next, prepare a training dataset to create a chatbot. Here, I randomly selected a topic with the aim of helping people resolve their uncertainties about which medical department to consult.

First, upload a PDF document, then ask Claude to organize 30 common questions from it.

![https://yushaing.vercel.app/assets/images/2-9d808e8ebe1cda3781c53460aa661a58.png](https://yushaing.vercel.app/assets/images/2-9d808e8ebe1cda3781c53460aa661a58.png)

### Specific Format Conversion[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#specific-format-conversion)

Visit the OpenAI Fine-tuning page to determine the specific format required for text content conversion. The text format is organized as follows:

![https://yushaing.vercel.app/assets/images/3-cc92d38502366cc7c93a8a98d848d0cf.png](https://yushaing.vercel.app/assets/images/3-cc92d38502366cc7c93a8a98d848d0cf.png)

The example demonstrates that data should be arranged in JSON (JavaScript Object Notation) format, which is convenient for exchanging, storing, and reading simple data. If you are not familiar with JSON, Claude can handle the formatting issues. Just copy the example format and modify it to the desired ChatGPT role; Claude will automatically generate the corresponding content.

![https://yushaing.vercel.app/assets/images/4-27ce2b6cd4eaa4bb56a971bb86ad3adb.png](https://yushaing.vercel.app/assets/images/4-27ce2b6cd4eaa4bb56a971bb86ad3adb.png)

With this, the preparation of the training dataset is complete. Next, upload this file to your personal GitHub, concluding the first step.

![https://yushaing.vercel.app/assets/images/5-1d298a9bb8bf3d7468ac8fa16dadeef5.png](https://yushaing.vercel.app/assets/images/5-1d298a9bb8bf3d7468ac8fa16dadeef5.png)

## **2. Writing Fine Tuning Code**[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#2-writing-fine-tuning-code)

First, open an editor, this time opting for vscode. Then create a new file named `fine_tune.py`.

**gpt-3.5_fine_tune.py**

```python
# gpt-3.5_fine_tune

#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import TextMessage, MessageEvent, TextSendMessage
import os
import openai
import tempfile
import datetime
import time
import string

import os

# Upgrade openai library
os.system('pip install openai --upgrade')

# Use curl to download the clinic_qa.json file
os.system('curl -o clinic_qa.json -L https://github.com/Hsiang-Tang/gpt-3.5_fine_tune/raw/main/clinic_qa.json')

import openai

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create fine-tune file
openai.File.create(
  file=open("clinic_qa.json", "rb"),
  purpose='fine-tune'
)

# List files
openai.File.list()

# Create fine-tuning job
openai.FineTuningJob.create(training_file="file-BhA5o5gmQRCx15KsK4zq97WI", model="gpt-3.5-turbo")

# List fine-tuning jobs
openai.FineTuningJob.list(limit=10)

# Retrieve fine-tuning job events
openai.FineTuningJob.retrieve("ftjob-x9NGckMPlxEXEXGSDtjy4Uc0")

# List fine-tuning job events
openai.FineTuningJob.list_events(id="ftjob-x9NGckMPlxEXEXGSDtjy4Uc0", limit=10)

# Create chat completion
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are now playing the role of a professional doctor"},
    {"role": "user", "content": "I feel tired when I move, always very fatigued"}
  ]
)

print(completion.choices[0].message.content)

# Create chat completion with fine-tuned model
completion2 = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:personal::7wllb3DZ",
  messages=[
    {"role": "system", "content": "You are now playing the role of a professional doctor"},
    {"role": "user", "content": "I feel tired when I move, always very fatigued"}
  ]
)

print(completion2.choices[0].message.content)

def GPT_response(text):
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0613:personal::7wllb3DZ",
        messages=[
            {"role": "system", "content": "You are now playing the role of a professional doctor"},
            {"role": "user", "content": text}
        ]
    )

    answer = response.choices[0].message.content

# Remove punctuation from the reply text
    answer = answer.translate(str.maketrans('', '', string.punctuation))

    return answer

```

### Environment Variable Settings[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#environment-variable-settings)

Environment Variable Settings Considering the requirements of cloud deployment, it's essential to avoid exposing the code containing the openai.api_key, as OpenAI would immediately delete the key if detected. Therefore, the openai.api_key is changed to `os.getenv("OPENAI_API_KEY")`, allowing the code to fetch the environment variable from the cloud platform. Additionally, the `Def GPT_response` code has been added, encapsulating `fine_tune.py` into a function for use by `app.py`. With this, step two is completed.

## **3. Developing the LineBot**[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#3-developing-the-linebot)

### Creating the app.py File[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#creating-the-apppy-file)

**app.py**

```python
from flask import Flask, render_template, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import TextMessage, MessageEvent, TextSendMessage
import os
import openai
import tempfile
import datetime
import time
import string
from fine_tune import GPT_response

app = Flask(__name__, template_folder='templates')
static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

# Channel Access Token
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))

# Channel Secret
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# Initialize OPENAI API Key settings
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route("/")
def index():
    return render_template("./index.html")

@app.route("/heroku_wake_up")
def heroku_wake_up():
    return "Hey! Wake Up!!"

# Listen to all Post Requests from /callback
@app.route("/callback", methods=['POST'])
def callback():
# get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
# get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
# handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# Handle messages
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text
    GPT_answer = GPT_response(msg)# Call the GPT_response function here to process the user's message
    print(GPT_answer)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(GPT_answer))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

```

## **4. Deploying the Application**[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#4-deploying-the-application)

The steps to begin deploying the LineBot on Render are as follows:

| Step | Description |
| --- | --- |
| 1 | Log into Render and connect to the GitHub Repository via Web Service. |
| 2 | In app.py, comment out @handle.add and from fine_tune import GPT_response. |
| 3 | First, upload app.py to GitHub for deployment. Note: Uploading fine_tune.py first might cause deployment failure due to the absence of the app module. |
| 4 | After successfully deploying app.py, upload fine_tune.py to GitHub to allow Render to deploy the new commit. |
| 5 | Once the above steps are completed, uncomment @handle.add and from fine_tune import GPT_response in app.py. |
| 6 | After the deployment of fine_tune.py is complete, check the Render log to ensure fine_tune.py has started running. Note: Ensure the correct order of steps to avoid errors. |
| 7 | Allow the entire system to run to completion, ensuring the fine-tuned chatbot functions properly. |
| 8 | Set up a Cronjob (https://console.cron-job.org/jobs) to call Render every 5 minutes, preventing Render from going idle for over 15 minutes, thus ensuring continuous responses from the chatbot. |

Let's look at the differences before and after fine-tuning:

![https://yushaing.vercel.app/assets/images/6-1b663c67f5f4d5190e08f8962563b441.png](https://yushaing.vercel.app/assets/images/6-1b663c67f5f4d5190e08f8962563b441.png)

### Comparison Before and After[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#comparison-before-and-after)

- Before Fine-Tuning: GPT's responses were more disorganized and often inaccurate.
- After Fine-Tuning: Responses have significantly improved, becoming more precise.

### Training Costs[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#training-costs)

- A single training session (30 dialogues) costs about $0.04, approximately 1.28 New Taiwan Dollars.
- OpenAI initially provided an $18 quota for early registrations, while new accounts currently have about a $5 quota, which is usually sufficient.

### Errors During Deployment[](https://yushaing.vercel.app/docs/Artificial%20Intelligence/LineBot+gpt-3.5_fine%20tuning#errors-during-deployment)

- Encountered several errors during deployment.
- Provided records of the errors and the debugging process for reference.

| Issue Number | Description |
| --- | --- |
| 1. Pip Package Issue | - Check the Render log to confirm the pip version is up to date.- Add os.system('pip install openai — upgrade') in the code. |
| 2. GPT API Key Issue | - Since Render connects to a public GitHub project, the API key would be deleted by OpenAI if exposed.- Set up environment variables in the Render backend, using os.getenv("OPENAI_API_KEY") in the code. |
| 3. Structural Issue with app.py and fine_tune.py | - Separate the app.py and fine_tune.py files.- In app.py, import and call the function Def GPT_response(text) from fine_tune.py. |
| 4. Deployment Order Issue | - First upload app.py, then upload fine_tune.py to GitHub after successful deployment. |
| 5. Code Auto-execution Issue | - Due to the addition of from fine_tune import GPT_response in the code, Render fails to find the module.- Initially comment out this line and re-upload to GitHub. |
| 6. Render log RateLimitError | - Comment out handle_message and from fine_tune import GPT_response.- Allow the application to process incoming requests first, then gradually introduce GPT_response. |
| 7. Render log Shows 404 Not Found | - Define a route in the code to match the /callback path, handling Webhook requests for Line Bot. |
| 8. Setting up a Cronjob | - Set up a call to Render every 5 minutes through https://console.cron-job.org/jobs to prevent Render from going idle. |
