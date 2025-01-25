from .nodes.comfyui_inodes import *

NODE_CLASS_MAPPINGS = {
    "IIfElse": IIfElse,
    "IMultilineSplit": IMultilineSplit,
    "IAzureAiApi": IAzureAiApi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IIfElse": "If-Else",
    "IMultilineSplit": "Multiline Split",
    "IAzureAiApi": "Azure AI API",
}