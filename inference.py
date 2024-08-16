# *
# * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# * SPDX-License-Identifier: MIT-0
# *
# * Permission is hereby granted, free of charge, to any person obtaining a copy of this
# * software and associated documentation files (the "Software"), to deal in the Software
# * without restriction, including without limitation the rights to use, copy, modify,
# * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# * permit persons to whom the Software is furnished to do so.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# *

import re
import base64
from io import BytesIO
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def model_fn(model_dir):
    # Load our model from Hugging Face
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

    # Move model to GPU
    model.to(device)

    return model, processor


def predict_fn(data, model_and_processor):
    # unpack model and tokenizer
    model, processor = model_and_processor
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    #image = data.get("image")
    imagetype = data['inputs']['image']
    image_data = base64.b64decode(imagetype)
    image = Image.open(BytesIO(image_data))
    print(data)
    question = data['question']
    print(question)
    pixel_values = processor.feature_extractor(image, return_tensors="pt").pixel_values
    print(pixel_values)
    prompt = task_prompt.replace("{user_input}", question)
    print(prompt)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = prediction.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    prediction = re.sub(r"<.*?>", "", prediction, count=1).strip()
    print(processor.token2json(prediction))
    return prediction

