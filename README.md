## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:

### DESIGN STEPS:

#### Step1: Install Required Libraries:
Install the necessary Python libraries such as transformers, torch, and gradio.

#### Step2: Load the NER Model:
Use the transformers library to load a fine-tuned BART model for NER, for example, dslim/bart-large-ner.

#### Step3: Prepare the Output Format:
Organize the model output so that entities and their labels can be clearly highlighted in the interface.

#### Step4: Build the Gradio Interface:
Create a user-friendly interface with a text input box and a highlighted output display using the Gradio framework.

#### Step5: Customize the Interface:
Add colors for each entity type (like red for people, blue for organizations, green for locations) and include a title and description.

### PROGRAM:
```python
import os
import io
import base64
import json
import requests
import gradio as gr

from IPython.display import Image, display, HTML
from PIL import Image
from dotenv import load_dotenv, find_dotenv

# Load .env file
_ = load_dotenv(find_dotenv())

# Hugging Face API key
hf_api_key = os.environ['HF_API_KEY']

# NER endpoint
API_URL = os.environ['HF_API_NER_BASE']

print("NER API URL:", API_URL)


# -----------------------------------
# Helper function
# -----------------------------------

def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": inputs
    }

    if parameters is not None:
        data.update({
            "parameters": parameters
        })

    response = requests.request(
        "POST",
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(data)
    )

    return json.loads(response.content.decode("utf-8"))


# -----------------------------------
# Merge tokens
# -----------------------------------

def merge_tokens(tokens):
    merged_tokens = []

    for token in tokens:

        if (
            merged_tokens
            and token['entity'].startswith('I-')
            and merged_tokens[-1]['entity'].endswith(token['entity'][2:])
        ):

            # Current token continues previous entity
            last_token = merged_tokens[-1]

            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (
                last_token['score'] + token['score']
            ) / 2

        else:
            # New entity
            merged_tokens.append(token)

    return merged_tokens


# -----------------------------------
# NER function
# -----------------------------------

def ner(input):
    output = get_completion(
        input,
        parameters=None,
        ENDPOINT_URL=API_URL
    )

    merged_tokens = merge_tokens(output)

    return {
        "text": input,
        "entities": merged_tokens
    }


# -----------------------------------
# Gradio interface
# -----------------------------------

gr.close_all()

demo = gr.Interface(
    fn=ner,

    inputs=[
        gr.Textbox(
            label="Text to find entities",
            lines=2
        )
    ],

    outputs=[
        gr.HighlightedText(
            label="Text with entities"
        )
    ],

    title="NER with dslim/bert-base-NER",

    description=(
        "Find entities using the "
        "`dslim/bert-base-NER` model under the hood!"
    ),

    allow_flagging="never",

    examples=[
        "My name is Andrew, I'm building DeeplearningAI and I live in California",
        "My name is Poli, I live in Vienna and work at HuggingFace"
    ]
)


# -----------------------------------
# Launch
# -----------------------------------

demo.launch(
    share=True,
    server_port=int(os.environ['PORT4'])
)

```
### OUTPUT:

<img width="1137" height="602" alt="Screenshot 2026-09-05 181502" src="https://github.com/user-attachments/assets/d09aec03-53ea-428f-a333-724330c29119" />

<img width="1156" height="612" alt="Screenshot 2026-09-05 181332" src="https://github.com/user-attachments/assets/a82a20cc-fc77-4d39-bb12-23f7ecdf41f8" />

<img width="1157" height="610" alt="Screenshot 2026-09-05 181430" src="https://github.com/user-attachments/assets/4c4a1229-3d04-4bcd-b85d-c0a55728c354" />



### RESULT:

Thus, development of a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation excuted successfully.
