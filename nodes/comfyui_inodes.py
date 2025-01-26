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
                "endpoint": ("STRING", {"default": ""}),
                "key": ("STRING", {"default": ""}),
                "api_version": ("STRING", {"default": ""}),
                "deployment_name": ("STRING", {"default": ""}),
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


import zipfile
from datetime import datetime
from enum import Enum
class ZippingMethod(Enum):
    BY_SIZE = "bySize"
    BY_COUNT = "byCount"
class IZipDirectoryCreator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "zipping_method": ("STRING", {"default": ZippingMethod.BY_SIZE.value, "enum": [method.value for method in ZippingMethod]}),
                "size_value(MB)": ("INT", {"default": 1024}),
                "count_value": ("INT", {"default": 100}),
                "output_filenamePattern": ("STRING", {"default": "output"}),
                "output_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_file_paths",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "execute"

    CATEGORY = "File Operations"

    def execute(self, directory, zipping_method, size_value, count_value, output_filenamePattern, output_path, **kwargs):
        # Change color to yellow
        print("\x1b[33m")

        print(f"Starting zip operation in directory: {directory}, zipping method: {zipping_method} with output path: {output_path}, output filename pattern: {output_filenamePattern}, size value: {size_value}, count value: {count_value}")
        if not os.path.isdir(directory):
            print(f"Directory does not exist: {directory}")
            raise ValueError(f"Directory does not exist: {directory}")

        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not files:
            print(f"No files found in directory: {directory}")
            raise ValueError(f"No files found in directory: {directory}")

        os.makedirs(output_path, exist_ok=True)
        print(f"Output path created: {output_path}")

        def create_zip_file(zip_files, index):
            zip_filename = f"{output_filenamePattern}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{index}.zip"
            zip_filepath = os.path.join(output_path, zip_filename)
            print(f"Creating zip file: {zip_filepath}")
            with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                for file in zip_files:
                    zipf.write(file, os.path.basename(file))
                    print(f"Added file to zip: {file}")
            return zip_filepath

        zip_files = []
        zip_file_paths = []
        if zipping_method == ZippingMethod.BY_SIZE.value:
            current_size = 0
            for file in files:
                file_size = os.path.getsize(file)
                if current_size + file_size > size_value * 1024 * 1024:
                    zip_file_paths.append(create_zip_file(zip_files, len(zip_file_paths)))
                    zip_files = []
                    current_size = 0
                zip_files.append(file)
                current_size += file_size
            if zip_files:
                zip_file_paths.append(create_zip_file(zip_files, len(zip_file_paths)))
        elif zipping_method == ZippingMethod.BY_COUNT.value:
            for i in range(0, len(files), count_value):
                zip_files = files[i:i + count_value]
                zip_file_paths.append(create_zip_file(zip_files, len(zip_file_paths)))
        else:
            print(f"Invalid zipping method: {zipping_method}")
            raise ValueError(f"Invalid zipping method: {zipping_method}")

        print(f"Zip operation completed. Created files: {zip_file_paths}")
        # Reset color
        print("\x1b[0m")
        return (", ".join(zip_file_paths),)