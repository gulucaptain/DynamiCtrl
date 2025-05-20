from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
 
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/user/ckpt/qwen", torch_dtype="auto", device_map="auto"
)
 
# default processer
processor = AutoProcessor.from_pretrained("/home/user/code/ckpt/qwen")
 
image_data_root = "/home/user/images"
images = os.listdir(image_data_root)
images.sort()

for image in tqdm(images):
    image_path = os.path.join(image_data_root, image)
    Question_description = "Describe the person and the background." #FIXME Replace the question you want.
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": Question_description},
            ],
        }
    ]
 
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    with open("/home/user/descriptions.txt", "a") as f:
        f.writelines(image+"#####"+output_text+"\n")
    