  
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_ollama import OllamaLLM
import requests

class IAzureAiApi:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "endpoint": ("STRING", {"default": "https://wallpaperai.openai.azure.com/"}),
                "key": ("STRING", {"default": ""}),
                "api_version": ("STRING", {"default": "2024-08-01-preview"}),
                "deployment_name": ("STRING", {"default": "gpt-4o"}),
                "max_tokens": ("INT", {"default": 1000}),
                "prompt": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "execute"

    CATEGORY = "API"

    def execute(self, endpoint, key, api_version, deployment_name, max_tokens, prompt, **kwargs):
        print("prompt: ", prompt)
        llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            max_tokens=max_tokens,
            api_key=key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

        try:
            response = llm.invoke(prompt)
            print(response)
            return (response.content,)
        except Exception as e:
            print(f"Failed to get chat completion: {str(e)}")
            raise ValueError(f"Failed to get chat completion: {str(e)}")
        
class ILoadAzureAiApi:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "endpoint": ("STRING", {"default": "https://wallpaperai.openai.azure.com/"}),
                "key": ("STRING", {"default": ""}),
                "api_version": ("STRING", {"default": "2024-08-01-preview"}),
                "deployment_name": ("STRING", {"default": "gpt-4o"}),
                "max_tokens": ("INT", {"default": 1000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "execute"

    CATEGORY = "API"

    def execute(self, endpoint, key, api_version, deployment_name, max_tokens, **kwargs):
        llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            max_tokens=max_tokens,
            api_key=key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

        try:
            return (llm,)
        except Exception as e:
            print(f"Failed to get chat completion: {str(e)}")
            raise ValueError(f"Failed to get chat completion: {str(e)}")
    
class ILLMExecute:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "model": ("MODEL", {"default": ""}),
                "prompt": ("STRING", {"default": ""}),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "execute"

    CATEGORY = "API"

    def execute(self, model, prompt, enable, **kwargs):
        print("prompt: ", prompt)
        if not enable:
            return (prompt,)
        try:
            response = model.invoke(prompt)
            print(response)
            if isinstance(response, str):
                return (response,)
            return (response.content,)
        except Exception as e:
            print(f"Failed to get chat completion: {str(e)}")
            raise ValueError(f"Failed to get chat completion: {str(e)}")

class ILLMExecute2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "first_model": ("MODEL", {"default": ""}),
                "prompt": ("STRING", {"default": ""}),
                "use_first_model": ("BOOLEAN", {"default": True}),
                "enable": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "second_model": ("MODEL", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "execute"

    CATEGORY = "API"

    def execute(self, first_model, prompt, use_first_model, second_model=None, enable=True, **kwargs):
        print("prompt: ", prompt)
        if not enable:
            return (prompt,)
        model = first_model if use_first_model or not second_model else second_model
        try:
            response = model.invoke(prompt)
            print(response)
            if isinstance(response, str):
                return (response,)
            return (response.content,)
        except Exception as e:
            print(f"Failed to get chat completion: {str(e)}")
            raise ValueError(f"Failed to get chat completion: {str(e)}")
        
class IOllama:
    def __init__(self, url, port, model):
        self.url = url
        self.port = port
        self.model = model
    
    def invoke(self, prompt):
        llm = OllamaLLM(model=self.model, base_url=f"{self.url}:{self.port}")
        response = llm(prompt)
        return response
    
class ILoadOllamaApi:
    @staticmethod
    def _fetch_models():
        print("Attempting to fetch Ollama models at startup...")
        try:
            url = "http://localhost"    # or wherever your Ollama is
            port = "10000"
            resp = requests.get(f"{url}:{port}/api/tags", timeout=2)
            data = resp.json()
            model_names = [model["name"] for model in data["models"]]
            # cls._ollama_models = model_names
            # print("Loaded Ollama models at startup:", cls._ollama_models)
            print("Loaded Ollama models at startup:", model_names)
            return model_names
        except Exception as e:
            print("Could not load Ollama models at startup:", e)
    
    _ollama_models = _fetch_models()
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "url": ("STRING", {"default": "http://"}),
                "port": ("STRING", {"default": "10000"}),
                "model": (ILoadOllamaApi._ollama_models, {}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "execute"

    CATEGORY = "API"

    def execute(self, url, port, model, **kwargs):
        llm = IOllama(url, port, model)

        try:
            return (llm,)
        except Exception as e:
            print(f"Failed to get chat completion: {str(e)}")
            raise ValueError(f"Failed to get chat completion: {str(e)}")
    
class IPostProcessLLMResponse:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "response": ("STRING", {"default": ""}),
                "word": ("STRING", {"default": "</think>"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "execute"

    CATEGORY = "Text Processing"

    def execute(self, response, word, **kwargs):
        response = response if isinstance(response, str) else response[0]
        if word in response:
            response = response.split(word)[1]
        
        return (response,)
    