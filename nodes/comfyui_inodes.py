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


class IMultilineSplit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "code": ("STRING", {
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

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("prompt_list", "prompt_strings")
    OUTPUT_IS_LIST = (False, True)

    FUNCTION = "execute"

    CATEGORY = "Text Processing"

    def execute(self, code, remove_empty_lines=True, ignore_lines_starting_with="", **kwargs):
        ignore_prefixes = ignore_lines_starting_with.splitlines()
        lines = code.splitlines()
        if remove_empty_lines:
            lines = [line for line in lines if len(line.strip()) > 0]
        if ignore_prefixes:
            lines = [line for line in lines if not any(line.startswith(prefix) for prefix in ignore_prefixes)]
        
        return (lines, lines)


class ITimes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "times": ("INT",),
            },
            "optional": {
                "string_input": ("STRING",),
                "list_input": ("LIST",),
            },
        }
        
    RETURN_TYPES = ("STRING")
    RETURN_NAMES = ("prompt_strings")
    OUTPUT_IS_LIST = (True)
    
    FUNCTION = "execute"
    
    CATEGORY = "Text Processing"
    
    def execute(self, times, string_input="", list_input=[], **kwargs):
        if string_input:
            return [string_input] * times
        elif list_input:
            return list_input * times
        else:
            return [""] * times
        
class IListCounter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list_input": ("LIST",),
            },
        }
        
    RETURN_TYPES = ("INT")
    RETURN_NAMES = ("count")
    
    FUNCTION = "execute"
    
    CATEGORY = "Text Processing"
    
    def execute(self, list_input, **kwargs):
        return (len(list_input),)
    
class ICutList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list_input": ("LIST",),
                "start": ("INT",),
                "end": ("INT",),
            },
        }
        
    RETURN_TYPES = ("LIST")
    RETURN_NAMES = ("list_output")
    
    FUNCTION = "execute"
    
    CATEGORY = "Text Processing"
    
    def execute(self, list_input, start, end, **kwargs):
        return (list_input[start:end],)
    
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
        

NODE_CLASS_MAPPINGS = {
    "IIfElse": IIfElse,
    "IMultilineSplit": IMultilineSplit,
    "IAzureAiApi": IAzureAiApi,
    "ITimes": ITimes,
    "IListCounter": IListCounter,
    "ICutList": ICutList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IIfElse": "I If-Else",
    "IMultilineSplit": "I Multiline Split",
    "IAzureAiApi": "I Azure AI API",
    "ITimes": "I Times",
    "IListCounter": "I List Counter",
    "ICutList": "I Cut List",
}