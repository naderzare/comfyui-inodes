import os
import folder_paths
import datetime

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
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            return {"dummy": str(datetime.datetime.now())}


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
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            return {"dummy": str(datetime.datetime.now())}

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
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            return {"dummy": str(datetime.datetime.now())}



class IStringsCounter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings": ("STRING",),
            },
        }

    # Return an integer for downstream usage
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "execute"

    CATEGORY = "Text Processing"
    
    def execute(self, strings, **kwargs):
        # `strings` will be a list of Python strings
        count = len(strings)

        # Here is the key: return a dict with both `result` (the int)
        # and a UI list with a text entry.
            
        return {"ui": {"text": str(count)}, "result": (count,), }
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            return {"dummy": str(datetime.datetime.now())}

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
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            return {"dummy": str(datetime.datetime.now())}

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
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            return {"dummy": str(datetime.datetime.now())}
    
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
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            return {"dummy": str(datetime.datetime.now())}

import os
from PIL import Image
import folder_paths
import numpy as np

import os
from PIL import Image
import folder_paths
import numpy as np

# class ISaveImage:
#     def __init__(self):
#         self.compress_level = 4

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "images": ("IMAGE", {"tooltip": "The images to save."}),
#                 "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save."})
#             },
#             "optional": {
#                 "path": ("STRING", {"default": "", "tooltip": "The path to save the images to."}),
#             }
#         }

#     INPUT_IS_LIST = (True,)
#     RETURN_TYPES = ("STRING",)  # Define a dummy output type
#     RETURN_NAMES = ("status",)  # Name the dummy output
#     FUNCTION = "save_images"
#     CATEGORY = "image"
#     DESCRIPTION = "Saves the input images to your ComfyUI output directory."

#     def save_images(self, images, filename_prefix="ComfyUI", path="", **kwargs):
#         # Generate the output folder and base filename
#         filename_prefix = filename_prefix[0] if isinstance(filename_prefix, list) else filename_prefix
#         path = path[0] if isinstance(path, list) else path
#         comfy_path = folder_paths.get_output_directory()
#         if not path:
#             path = comfy_path
#         else:
#             path = os.path.join(comfy_path, path)
#         os.makedirs(path, exist_ok=True)

#         for index, image in enumerate(images):
#             # Convert the image tensor to a PIL image
#             image_array = 255. * image.cpu().numpy()
#             img = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

#             # Create the filename for the image
#             filename = f"{filename_prefix}_{index:05}.png"
#             file_path = os.path.join(path, filename)

#             # Save the image
#             img.save(file_path, compress_level=self.compress_level)

#         # Return a status message
#         return (f"Images saved successfully to {path}",)
#     @classmethod
#     def IS_CHANGED(cls, *args, **kwargs):
#             #always update
#             return {"dummy": str(datetime.datetime.now())}

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
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print(args)
            print(kwargs)
            return {"dummy": str(datetime.datetime.now())}
        
    
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
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            return {"dummy": str(datetime.datetime.now())}
    
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

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
            #always update
            return {"dummy": str(datetime.datetime.now())}
        
class ISaveImage:
    counter = 0
    def __init__(self):
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "path": ("STRING", {"default": ""}),
                "show_all": ("BOOLEAN", {"default": False}),
                "show_count": ("INT", {"default": 1}),
                "show_preview": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    # We'll now return one string, which is the path, so next nodes can receive it.
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory, shows optional previews, and returns the path for the next node."
    
    def get_file_name(self, filename_prefix, path):
        files_in_dir = os.listdir(path)
        new_filename = f"{filename_prefix}_{ISaveImage.counter}.png"
        while new_filename in files_in_dir:
            ISaveImage.counter += 1
            new_filename = f"{filename_prefix}_{ISaveImage.counter}.png"
        ISaveImage.counter += 1
        return new_filename

    def save_images(self, images, filename_prefix="ComfyUI", path="", show_all=False, show_count=1, show_preview=False, **kwargs):
        show_all = show_all[0] if isinstance(show_all, list) else show_all
        show_count = show_count[0] if isinstance(show_count, list) else show_count
        show_preview = show_preview[0] if isinstance(show_preview, list) else show_preview
        comfy_path = folder_paths.get_output_directory()
        path = path[0] if isinstance(path, list) else path
        opath = os.path.join(comfy_path, path) if path else comfy_path
        os.makedirs(opath, exist_ok=True)
        filename_prefix = filename_prefix[0] if isinstance(filename_prefix, list) else filename_prefix
        
        results = []
        for (batch_number, image) in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            # If you have extra PNG metadata, you could pass it with pnginfo
            file = self.get_file_name(filename_prefix, opath)
            img.save(os.path.join(opath, file), compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": opath,  # the final directory path
                "type": self.type
            })

        # Return a dictionary where:
        # 1) 'ui': { 'images': ... } is optional, controlling what is displayed in the UI.
        # 2) 'saved_path' is the string that goes to the next node.
        if not show_preview:
            return {"saved_path": opath}
        if show_all:
            return {"saved_path": opath, "ui": {"images": results}}
        else:
            return {"saved_path": opath, "ui": {"images": results[:show_count]}}


class ISaveText:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
                "filename": ("STRING",),
                "path": ("STRING",),
            },
        }
        
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "Text Processing"
    
    def execute(self, text, filename, path, **kwargs):
        text = text[0] if isinstance(text, list) else text
        filename = filename[0] if isinstance(filename, list) else filename
        path = path[0] if isinstance(path, list) else path
        comfy_path = folder_paths.get_output_directory()
        if not path:
            path = comfy_path
        else:
            path = os.path.join(comfy_path, path)
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        with open(filepath, "w") as f:
            f.write(text)
        return (filepath,)
    
import os
import zipfile
import math

class IZipImages:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "files_count": ("INT", {"default": 100}),
                "files_size": ("FLOAT", {"default": 100}),
                "by_count": ("BOOLEAN", {"default": True}),
                "signal": ("STRING", {"default": ""}),
            },
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "zip_images_in_chunks"
    OUTPUT_NODE = True
    CATEGORY = "File Operations"

    def zip_images_in_chunks(self, path, files_count, files_size, by_count, signal=None, **kwargs):
        """
        Zips image files (.png, .jpg, .jpeg, etc.) from a given path into multiple zip archives.

        If by_count is True, each zip file will contain up to `files_count` images.
        If by_count is False, each zip file will contain images until their total size in MB reaches `files_size`.

        :param path: The directory path containing images.
        :param files_count: The number of files per zip if by_count is True.
        :param files_size: The size in MB for each zip file if by_count is False.
        :param by_count: Boolean indicating whether to split by file count or by file size.
        """
        comfy_path = folder_paths.get_output_directory()
        path = path[0] if isinstance(path, list) else path
        files_count = files_count[0] if isinstance(files_count, list) else files_count
        files_size = files_size[0] if isinstance(files_size, list) else files_size
        by_count = by_count[0] if isinstance(by_count, list) else by_count
        
        if not path:
            path = comfy_path
        else:
            path = os.path.join(comfy_path, path)

        # Gather all image files (case-insensitive match of .jpg or .png)
        valid_ext = {'.png', '.jpg', '.jpeg'}
        image_files = []

        # Traverse the directory
        for root, dirs, files in os.walk(path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_ext:
                    image_files.append(os.path.join(root, file))

        # Sort images by name (optional, but consistent)
        image_files.sort()

        if not image_files:
            print("No image files found.")
            return (None,)

        zip_index = 1
        current_zip_file = None
        current_zip = None

        def start_new_zip(index):
            # Close old zip if it exists
            if current_zip:
                current_zip.close()
            zip_filename = os.path.join(path, f"images_part_{index}.zip")
            return zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)

        # Initialize first zip
        current_zip = start_new_zip(zip_index)

        # Trackers
        file_count_in_zip = 0
        current_size_mb = 0.0

        for image_path in image_files:
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024.0)

            if by_count:
                # If adding another file exceeds the files_count limit, start a new zip
                if file_count_in_zip >= files_count:
                    current_zip.close()
                    zip_index += 1
                    current_zip = start_new_zip(zip_index)
                    file_count_in_zip = 0
                    current_size_mb = 0.0
            else:
                # If adding this file exceeds the files_size limit, start a new zip
                if current_size_mb + file_size_mb > files_size:
                    current_zip.close()
                    zip_index += 1
                    current_zip = start_new_zip(zip_index)
                    file_count_in_zip = 0
                    current_size_mb = 0.0

            # Add file to current zip
            arcname = os.path.relpath(image_path, start=path)
            current_zip.write(image_path, arcname)

            # Update trackers
            file_count_in_zip += 1
            current_size_mb += file_size_mb

        # Close the last zip
        if current_zip:
            current_zip.close()

        print(f"Zipping complete. Created {zip_index} zip file(s).")
        
        return (path,)


class IMergeImages:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "in_prefix": ("STRING", {"default": "image"}),
                "out_prefix": ("STRING", {"default": "merged"}),
                "count_on_x": ("INT", {"default": 5}),
                "requested_height": ("INT", {"default": 1800}),
                "requested_width": ("INT", {"default": 1800}),
                "times": ("INT", {"default": 1}),
            },
        }
        
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("saved_path", "out_prefix")
    OUTPUT_IS_LIST = (False, False,)
    FUNCTION = "merge_images"
    OUTPUT_NODE = False
    CATEGORY = "Image Processing"
    
    def merge_images(self, path, in_prefix, out_prefix, count_on_x, requested_height, requested_width, times, **kwargs):
        for counter in range(times):
            valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
            all_images = [
                os.path.join(path, f) for f in os.listdir(path)
                if os.path.splitext(f.lower())[-1] in valid_extensions and f.startswith(in_prefix)
            ]
            
            if not all_images:
                raise ValueError(f"No valid images found in directory: {path}")

            # --- Randomly pick ONE image to get base dimensions (w, h) ---
            sample_image_path = random.choice(all_images)
            with Image.open(sample_image_path) as img_sample:
                w, h = img_sample.size  # original dimensions of the image
            
            # --- 2-4. Compute new_w, t, new_h ---
            new_w = requested_width / float(count_on_x)
            t = new_w / float(w)
            new_h = t * float(h)
            
            # --- 5. Compute count_on_y ---
            #     We use floor division for integer math, then +1
            count_on_y = int(requested_height // new_h) + 1
            
            # --- 6. Calculate how many total images we need ---
            r = count_on_x * count_on_y
            
            if len(all_images) < r:
                all_images = all_images * math.ceil(r / len(all_images)) 
            
            # Randomly pick r images (no repetition). 
            # If you want duplicates possible, you could do something else.
            chosen_images = random.sample(all_images, r)
            
            # --- 7. Create the final mosaic image ---
            final_mosaic_width = int(new_w) * count_on_x
            final_mosaic_height = int(new_h) * count_on_y

            mosaic_image = Image.new('RGB', (final_mosaic_width, final_mosaic_height))
            
            # --- Paste images row by row ---
            idx = 0
            for y in range(count_on_y):
                for x in range(count_on_x):
                    img_path = chosen_images[idx]
                    idx += 1
                    
                    # Open and resize
                    with Image.open(img_path) as im:
                        im_resized = im.resize(
                            (int(new_w), int(new_h)),
                            resample=Image.Resampling.LANCZOS
                        )
                    
                    # Compute top-left corner for this tile
                    left = x * int(new_w)
                    top = y * int(new_h)
                    
                    # Paste the tile
                    mosaic_image.paste(im_resized, (left, top))
            
            # --- Save the final mosaic ---
            output_path = os.path.join(path, f"{out_prefix}_{counter}.jpg")
            counter += 1
            mosaic_image.save(output_path)
            print(f"Saved mosaic to '{output_path}'.")
            
        return (path, out_prefix,)
    

class IFinalizeProject:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "anything": (any_type, {}), 
                "except_prefix": ("STRING", {"default": ""}),
            },
        }
        
    RETURN_TYPES = ("STRING", "STRING")
    OUTPUT_IS_LIST = (False, False)
    RETURN_NAMES = ("zip_path", "saved_path",)
    FUNCTION = "finalize_project"
    OUTPUT_NODE = False
    CATEGORY = "File Operations"
    
    def finalize_project(self, path, **kwargs):
        # 
        path = path[0] if isinstance(path, list) else path
        comfy_path = folder_paths.get_output_directory()
        except_prefix = kwargs.get("except_prefix", "")
        except_prefix = except_prefix[0] if isinstance(except_prefix, list) else except_prefix
        
        # remove all images in path except prefix
        image_postfix = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        for f in os.listdir(path):
            if f.startswith(except_prefix):
                continue
            if os.path.splitext(f.lower())[-1] in image_postfix:
                os.remove(os.path.join(path, f))
    
        # zip the path and put it into comfy_path
        project_name = os.path.basename(path)
        zip_filename = os.path.join(comfy_path, f"{project_name}.zip")
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, dirs, files in os.walk(path):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path))
        
        return (zip_filename, path)
    
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from datetime import datetime

class IUploadToGoogleDrive:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "parent_folder_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "service_account_file_path": ("STRING", {"default": ""}),
                "service_account_json": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": "merged"}),
                "suffix": ("STRING", {"default": "zip,txt"}),
            },
        }
        
    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "upload_to_google_drive"
    OUTPUT_NODE = False
    CATEGORY = "File Operations"
    
    def upload_to_google_drive(self, path, parent_folder_id, **kwargs):
        print("Upload to Google Drive")
        path = path[0] if isinstance(path, list) else path
        parent_folder_id = parent_folder_id[0] if isinstance(parent_folder_id, list) else parent_folder_id
        service_account_file_path = kwargs.get("service_account_file_path", "")
        service_account_file_path = service_account_file_path[0] if isinstance(service_account_file_path, list) else service_account_file_path
        service_account_json = kwargs.get("service_account_json", "")
        service_account_json = service_account_json[0] if isinstance(service_account_json, list) else service_account_json
        prefix = kwargs.get("prefix", "")
        prefix = prefix[0] if isinstance(prefix, list) else prefix
        suffix = kwargs.get("suffix", "")
        suffix = suffix[0] if isinstance(suffix, list) else suffix
        
        if len(prefix) == 0 and len(suffix) == 0:
            raise ValueError("Please provide either prefix or suffix.")
        
        prefix = prefix.split(",")
        suffix = suffix.split(",")
        
        print("path", path)
        print("parent_folder_id", parent_folder_id)
        print("service_account_file_path", service_account_file_path)
        print("service_account_json", service_account_json)
        print("prifix", prefix)
        print("suffix", suffix)
        
        if not service_account_file_path and not service_account_json:
            raise ValueError("Please provide either service_account_file_path or service_account_json.")
        
        if service_account_json:
            comfui_path = folder_paths.get_output_directory()
            service_account_file_path = os.path.join(comfui_path, "service_account.json")
            with open(service_account_file_path, "w") as f:
                f.write(service_account_json)
            print(f"Service account JSON written to {service_account_file_path}")

        SCOPES = ['https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file_path, scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=credentials)

        new_folder_name = os.path.basename(path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_folder_name = f"{current_time}_{new_folder_name}"
        local_directory_path = path
        # 2. Create a new folder in Google Drive under the parent folder
        folder_metadata = {
            'name': new_folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        created_folder = service.files().create(body=folder_metadata, fields='id').execute()
        new_folder_id = created_folder.get('id')
        print(f"Created new folder '{new_folder_name}' with ID: {new_folder_id}")

        # 3. Find all .zip files in the local directory
        files = []
        for file in os.listdir(local_directory_path):
            prifix_match = any([file.startswith(p) for p in prefix])
            suffix_match = any([file.endswith(s) for s in suffix])
            print(file, prifix_match, suffix_match)
            if prifix_match or suffix_match:
                files.append(file)

        print(f"Found {len(files)} files to upload")
        print(files)
        # 4. Upload each zip file to the newly created folder
        image_suffix = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        for file in files:
            file_path = os.path.join(local_directory_path, file)
            if file.endswith(".zip"):
                media = MediaFileUpload(file_path, mimetype='application/zip', resumable=True)
            if file.endswith(".txt"):
                media = MediaFileUpload(file_path, mimetype='text/plain', resumable=True)
            if os.path.splitext(file.lower())[-1] in image_suffix:
                media = MediaFileUpload(file_path, mimetype='image/jpeg', resumable=True)

            file_metadata = {
                'name': file,
                'parents': [new_folder_id]
            }

            upload_response = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            uploaded_file_id = upload_response.get('id')
            print(f"Uploaded '{file}' to folder ID {new_folder_id} with file ID: {uploaded_file_id}")
            
        return (f"Uploaded {len(files)} files to Google Drive folder {new_folder_id}",)
            