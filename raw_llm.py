import requests
from ollama import chat

def prompt_with_ollama(prompt, temp, num_pred):
    response = chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a software architect"},
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": temp,
            "num_predictions": num_pred,
            "stream": False
        }
    )
    return response['message']['content']

def prompt_with_requests(prompt, temp, num_pred):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "mistral",
        "messages": [
            {"role": "system", "content": "You are a software architect"},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": temp,
            "num_predictions": num_pred
        }
    }

    response = requests.post(url, json=payload)

    response.raise_for_status()  

    data = response.json()

    return data["message"]["content"]

for temp in [0, 0.3, 0.7, 1]:
    for num_pred in [50, 100, 200]:
        print(f"Temperature: {temp}, Num Predictions: {num_pred}")
        print("Using ollama library:")
        print(prompt_with_ollama("How do I create a REST API?", temp, num_pred))
        print("Using requests library:")
        print(prompt_with_requests("How do I create a REST API?", temp, num_pred))