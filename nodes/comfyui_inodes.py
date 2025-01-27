import os
import folder_paths

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False
any_type = AlwaysEqualProxy("*")

class IIfElse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (any_type,),
                "boolean": ("BOOLEAN",),
            },
        }

    RETURN_TYPES = (any_type, any_type,)
    RETURN_NAMES = ("if_true", "if_false",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic"

    def execute(self, input, boolean, **kwargs):
        if boolean:
            return (input, "")
        else:
            return ("", input)


class IMultilineSplitToStrings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "multiline": ("STRING", {
                    "default": "# Write your multi-line code here\nprint(\"Hello, World!\")\nresult = x + y",
                    "multiline": True,
                })
            },
            "optional": {
                "remove_empty_lines": ("BOOLEAN", {
                    "default": True
                }),
                "ignore_lines_starting_with": ("STRING", {
                    "default": "#",
                    "multiline": True
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("strings",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "execute"

    CATEGORY = "Text Processing"

    def execute(self, multiline, remove_empty_lines=True, ignore_lines_starting_with="", **kwargs):
        ignore_prefixes = ignore_lines_starting_with.splitlines()
        lines = multiline.splitlines()
        if remove_empty_lines:
            lines = [line for line in lines if len(line.strip()) > 0]
        if ignore_prefixes:
            lines = [line for line in lines if not any(line.startswith(prefix) for prefix in ignore_prefixes)]
        
        return (lines,)

class ITimesToStrings:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "times": ("INT", {"default": 1}),
            },
            "optional": {
                "strings": ("STRING", {"default": ""}),
            },
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("strings",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    
    CATEGORY = "Text Processing"
    
    def execute(self, times, strings=[]):
        print("string_input: ", strings)
        print(type(strings))
        times = times[0] if isinstance(times, list) else times
        if strings:
            new_list = []
            for i in range(times):
                print("-------------------------")
                for string in strings:
                    print(string)
                    new_list.append(string)
            print(len(new_list))
            return (new_list,)
        else:
            return ([""] * times,)


class IStringsCounter:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strings": ("STRING",),
            },
        }
        
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "execute"
    
    CATEGORY = "Text Processing"
    
    def execute(self, strings, **kwargs):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(type(strings))
        print(type(strings[0]))
        print(strings)
        count = len(strings)
        # Return both the UI display and the actual result
        return {"ui": {"text": f"Count: {count}"}, "result": (count,)}

class ICutStrings:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strings": ("STRING",),
                "start": ("INT",),
                "end": ("INT",),
                "is_enable": ("BOOLEAN", {"default": True}),
            },
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("list_output",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    
    CATEGORY = "Text Processing"
    
    def execute(self, strings, start, end, is_enable, **kwargs):
        if not is_enable[0]:
            return (strings[:],)
        return (strings[start[0]:end[0]],)

import random

class IRandomChoiceToStrings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings": ("STRING", {"default": []}),
                "n": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("random_choices",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    CATEGORY = "List Operations"

    def execute(self, strings, n, **kwargs):
        n = n[0] if isinstance(n, list) else n
        if n > len(strings):
            return (strings[:],)
        return (random.sample(strings, n),)
    
class IStringsToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings": ("STRING", {"default": ""}),
            },
            "optional": {
                "separator": ("STRING", {"default": ""}),
                "separator_line": ("BOOLEAN", {"default": False}),
            },
        }

    INPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_output",)
    FUNCTION = "execute"

    CATEGORY = "Text Processing"

    def execute(self, strings, separator, separator_line, **kwargs):
        separator = separator[0] if isinstance(separator, list) else separator
        if separator_line:
            res = f"\n".join(strings)
        else:
            res = f"{separator}".join(strings)
        return (res,)

import os
from PIL import Image
import folder_paths
import numpy as np

import os
from PIL import Image
import folder_paths
import numpy as np

class ISaveImage:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save."})
            },
            "optional": {
                "path": ("STRING", {"default": "", "tooltip": "The path to save the images to."}),
            }
        }

    INPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)  # Define a dummy output type
    RETURN_NAMES = ("status",)  # Name the dummy output
    FUNCTION = "save_images"
    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI", path="", **kwargs):
        # Generate the output folder and base filename
        filename_prefix = filename_prefix[0] if isinstance(filename_prefix, list) else filename_prefix
        path = path[0] if isinstance(path, list) else path
        comfy_path = folder_paths.get_output_directory()
        if not path:
            path = comfy_path
        else:
            path = os.path.join(comfy_path, path)
        os.makedirs(path, exist_ok=True)

        for index, image in enumerate(images):
            # Convert the image tensor to a PIL image
            image_array = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

            # Create the filename for the image
            filename = f"{filename_prefix}_{index:05}.png"
            file_path = os.path.join(path, filename)

            # Save the image
            img.save(file_path, compress_level=self.compress_level)

        # Return a status message
        return (f"Images saved successfully to {path}",)

import torch

class IPassImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "count": ("INT", {"default": 1}),
                "pass_all": ("BOOLEAN", {"default": True,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)  # Name the dummy output
    FUNCTION = "pass_images"
    CATEGORY = "image"
    DESCRIPTION = "Pass the input images to the output."
    INPUT_IS_LIST = (True, False, False,)
    OUTPUT_IS_LIST = (True,)

    def pass_images(self, images, count, pass_all, **kwargs):
        print("$$$$$$$ Pass Image")
        count = count[0] if isinstance(count, list) else count
        pass_all = pass_all[0] if isinstance(pass_all, list) else pass_all
        
        print(count)
        print(pass_all)
        print(len(images))
        print(images[0].shape)
        
        if pass_all:
            return (images[:],)
        images = torch.cat(images, dim=0)
        print(images.shape)
        # images = list(torch.unbind(images, dim=0))
        images = [img.unsqueeze(0) for img in torch.unbind(images, dim=0)]
        print(len(images))
        print(images[0].shape)

        if count < 1:
            count = 1
        return (images[:count],)
        
        
    
class IStringsToFile:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": "file.txt"}),
            },
            "optional": {
                "path": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "append": ("BOOLEAN", {"default": False}),
            }
        }

    INPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "execute"

    CATEGORY = "Text Processing"

    def execute(self, strings, filename, path, overwrite, append, **kwargs):
        path = path[0] if isinstance(path, list) else path
        filename = filename[0] if isinstance(filename, list) else filename
        comfy_path = folder_paths.get_output_directory()
        print("path: ", path)
        print("filename: ", filename)
        print("comfy_path: ", comfy_path)
        if not path:
            path = comfy_path
        else:
            path = os.path.join(comfy_path, path)
        if not filename:
            filename = "res"
        if not filename.endswith(".txt"):
            filename += ".txt"
        if not os.path.exists(path):
            os.makedirs(path)
        print("path: ", path)
        print("filename: ", filename)
        print("comfy_path: ", comfy_path)
        filepath = os.path.join(path, filename)
        print("filename: ", filename)
        mode = "w" if overwrite else "a" if append else "w"
        with open(filepath, mode) as f:
            f.write('\n'.join(strings))
        return (filepath,)
    
class IPromptGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "You are a prompt generator for creating photos. Write prompts based on the following idea, with each prompt limited to one line. I will use the prompts you generate to create a poster.",
                    "multiline": True,
                    }),
                "idea": ("STRING", {"default": "Idea: A red car"}),
                "count_str": ("STRING", {"default": "Generate 5 prompts"}),
                "times": ("INT", {"default": 1}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    
    CATEGORY = "Text Processing"
    
    def execute(self, prompt, idea, count_str, times=1, **kwargs):
        print("----------------------")
        print("prompt: ", prompt)
        print("idea: ", idea)
        print("count: ", count_str)
        print("times: ", times)
        prompt = prompt if isinstance(prompt, str) else prompt[0]
        idea = idea if isinstance(idea, str) else idea[0]
        count_str = count_str if isinstance(count_str, str) else count_str[0]
        res = f'''{prompt}\n{idea}\n{count_str}\n'''
        res = [res] * times
        return (res,)

    
from langchain_openai import AzureChatOpenAI


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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "execute"

    CATEGORY = "API"

    def execute(self, model, prompt, **kwargs):
        print("prompt: ", prompt)
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

    def execute(self, first_model, prompt, use_first_model, second_model=None, **kwargs):
        print("prompt: ", prompt)
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
        
from langchain_openai import AzureChatOpenAI
from langchain_ollama import OllamaLLM
import requests
    
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
    
    # _ollama_models = ["<No Models Found>"]

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
    