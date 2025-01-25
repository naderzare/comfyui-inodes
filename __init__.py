from .nodes.comfyui_inodes import *

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