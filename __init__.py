from .nodes.comfyui_inodes import *

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "IIfElse": IIfElse,
    "IMultilineSplitToStrings": IMultilineSplitToStrings,
    "IAzureAiApi": IAzureAiApi,
    "ITimesToStrings": ITimesToStrings,
    "IStringsCounter": IStringsCounter,
    "ICutStrings": ICutStrings,
    "IRandomChoiceToStrings": IRandomChoiceToStrings,
    "IStringsToString": IStringsToString,
    "IStringsToFile": IStringsToFile,
    "ISaveImage": ISaveImage,
    "IPromptGenerator": IPromptGenerator,
    "IPostProcessLLMResponse": IPostProcessLLMResponse,
    "IPassImage": IPassImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IIfElse": "I If Else",
    "IMultilineSplitToStrings": "I Multiline Split to Strings",
    "IAzureAiApi": "I Azure AI API",
    "ITimesToStrings": "I Times to Strings",
    "IStringsCounter": "I Strings Counter",
    "ICutStrings": "I Cut Strings",
    "IRandomChoiceToStrings": "I Random Choice to Strings",
    "IStringsToString": "I Strings to String",
    "IStringsToFile": "I Strings to File",
    "ISaveImage": "I Save Image",
    "IPromptGenerator": "I Prompt Generator",
    "IPostProcessLLMResponse": "I Post Process LLM Response",
    "IPassImage": "I Pass Image",
}
