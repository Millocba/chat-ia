import requests

def test_ollama():
    url = "http://localhost:11434/v1/completions"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "llama3.2:latest",
        "prompt": "¿Qué es el uso de la fuerza en el contexto policial?",  # Aquí puedes poner la pregunta o el contexto de tu elección
        "max_tokens": 150,
    }
    
    response = requests.post(url, json=data, headers=headers)
    print("Status Code:", response.status_code)
    print("Response Text:", response.json())

test_ollama()
