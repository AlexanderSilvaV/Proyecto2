from langchain_community.llms import Ollama

# Función que recibe el nombre del modelo y el prompt
def send_prompt(prompt: str, model_name: str) -> str:
    try:
        llm = Ollama(model=model_name, base_url="http://localhost:11434", verbose=True)
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error al invocar el modelo: {e}"