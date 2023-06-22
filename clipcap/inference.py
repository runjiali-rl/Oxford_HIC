import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    MaxLengthCriteria,
    StoppingCriteriaList
)
import skimage.io as io
import PIL.Image
from tqdm import tqdm
import cog
import pandas as pd
import inspect

# import torch

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

WEIGHTS_PATHS = {
    "coco": "coco_weights.pt",
    "conceptual-captions": "conceptual_weights.pt",
}

D = torch.device
CPU = torch.device("cpu")

#
# class Predictor(cog.Predictor):
#     def setup(self):
#         """Load the model into memory to make running multiple predictions efficient"""
#         self.device = torch.device("cuda")
#         self.clip_model, self.preprocess = clip.load(
#             "ViT-B/32", device=self.device, jit=False
#         )
#         self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
#         self.models = {}
#         self.prefix_length = 10
#         for key, weights_path in WEIGHTS_PATHS.items():
#             model = ClipCaptionModel(self.prefix_length)
#             model.load_state_dict(torch.load(weights_path, map_location=CPU))
#             model = model.eval()
#             model = model.to(self.device)
#             self.models[key] = model
#
#     @cog.input("image", type=cog.Path, help="Input image")
#     @cog.input(
#         "model",
#         type=str,
#         options=WEIGHTS_PATHS.keys(),
#         default="coco",
#         help="Model to use",
#     )
#     @cog.input(
#         "use_beam_search",
#         type=bool,
#         default=False,
#         help="Whether to apply beam search to generate the output text",
#     )
#     def predict(self, image, model, use_beam_search):
#         """Run a single prediction on the model"""
#         self.device = torch.device("cuda")
#         self.clip_model, self.preprocess = clip.load(
#             "ViT-B/32", device=self.device, jit=False
#         )
#         self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
#         self.models = {}
#         self.prefix_length = 10
#         image = io.imread(image)
#         model = self.models[model]
#         pil_image = PIL.Image.fromarray(image)
#         image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             prefix = self.clip_model.encode_image(image).to(
#                 self.device, dtype=torch.float32
#             )
#             prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
#         if use_beam_search:
#             return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
#         else:
#             return generate2(model, self.tokenizer, embed=prefix_embed)


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt2.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt2(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        out = self.scale.exp() * out.logits
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt2.transformer.wte.weight.shape[1]
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        for param in self.gpt2.parameters():
            param.requires_grad = False
        # self.gpt.eval()
        return self


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=20,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    # !!!!!!!!!!!!!!!!!!! start: modified vs original, to pass inputs_embeds when they are available
    if "inputs_embeds" in kwargs and past is None:  # we only want to use them in the 1st generation step
        inputs_embeds = kwargs.get("inputs_embeds", None)
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs.update({
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    return model_inputs


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search
        kwargs:
            Additional logits processor specific kwargs.

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores

def predict(image, model, prompt=None):
    """Run a single prediction on the model"""
    device = torch.device("cuda:2")
    clip_model, preprocess = clip.load(
        "ViT-B/32", device=device, jit=False
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_length = 10
    image = io.imread(image)
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(
            device, dtype=torch.float32
        )
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    model.gpt2.prepare_inputs_for_generation = prepare_inputs_for_generation
    input_ids = torch.LongTensor([[model.gpt2.config.bos_token_id]])
    num_beams = model.gpt2.config.num_beams

    num_beam_groups =  model.gpt2.config.num_beam_groups


    logits_processor = LogitsProcessorList()


    eos_token_id = model.gpt2.config.eos_token_id

    if eos_token_id is None and hasattr(model.gpt.config, "decoder"):
        eos_token_id = model.gpt2.config.decoder.eos_token_id


    max_length = model.gpt2.config.max_length
    min_length = model.gpt2.config.min_length

    input_ids_seq_length = input_ids.shape[-1]

    if min_length is not None and min_length > max_length:
        raise ValueError(
            f"Unfeasible length constraints: the minimum length ({min_length}) is larger than the maximum "
            f"length ({max_length})"
        )

    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=100)])
    logits_processor = model.gpt2._get_logits_processor(
            repetition_penalty=None,
            no_repeat_ngram_size=2,
            encoder_no_repeat_ngram_size=None,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=None,
            bad_words_ids=None,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            prefix_allowed_tokens_fn=None,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=None,
            remove_invalid_values=None,
            exponential_decay_length_penalty=None,
            logits_processor=logits_processor,
            renormalize_logits=None,
            suppress_tokens=None,
            begin_suppress_tokens=None,
            forced_decoder_ids=None,
        )

    if prompt is not None:
        prompt_input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        prompt_embedding = model.gpt2.transformer.wte(prompt_input_ids.to(device))
        prefix_embed = torch.cat((prefix_embed, prompt_embedding), dim=1)
    outputs = model.gpt2.greedy_search(
        input_ids.to(device), inputs_embeds=prefix_embed.to(device),
        stopping_criteria=stopping_criteria, pad_token_id=model.gpt2.config.eos_token_id,
        logits_processor=logits_processor
    )
    print(outputs)
    if prompt is not None:
        print("\ngreedy + inputs_embeds:", prompt, tokenizer.decode(outputs[0], skip_special_tokens=True))
    else:
        print("\ngreedy + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))


def main(model_path):

    model = ClipCaptionModel(prefix_length=10)
    model.load_state_dict(torch.load(model_path))
    model = model.to(torch.device('cuda:2'))

    # image_id = 'wisdom-cat'
    # image_path = '../datasets/images'
    # image = os.path.join(image_path, f'{image_id}.jpg')
    image = '../images/dog_demo.jpg'
    predict(image, model)
    # print(result)


if __name__ == '__main__':
    model_path = 'clipcap_humor_demo1.pth'
    save_path = 'clipcap_humor_mask_in_val_output.csv'
    image_id_path = '../datasets/in_val_ids.npy'
    main(model_path)
