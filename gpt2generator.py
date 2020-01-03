import os
from pathlib import Path
import itertools
import torch
import torch.nn.functional as F
import re
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from getconfig import settings, logger
from utils import cut_trailing_sentence

MIN_GENERATION_LENGTH=4
NUM_SAMPLES=100
#NOTE TODO Converts a sequence of tokens (string) in a single string. The most simple way to do it is ‘ ‘.join(self.convert_ids_to_tokens(token_ids)) but we often want to remove sub-word tokenization artifacts at the same time.

#This is a functional mess.
#Lets fix this by making multiple generators that share a single model.


# warnings.filterwarnings("ignore")
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}

def getTokens(tokenizer, l):
    tokenizer.encode()

#the tokenizer does not preserve white space at the front of the string.
#so we will append something else to the front of the string and then remove it after tokenization
def hackyEncode(tokenizer, s):
    return tokenizer.encode('====\n '+s)[2:]
    

def hackyWhiteSpaceCutter(prompt):
   return re.search(r'\s*$', prompt).group(0)

def memory_merge(prompt, context, tokenizer, maxHistory=1024):
        assert(prompt+context)
        #print(prompt+context)
        #logger.debug('RAW TEXT INPUT IS:`%r`', context)
        #the tokenizer is kind of broken for the first input, especially if it includes white space. Same with any trailing white space on the last output.
        #I'm going with the add prefix option but I'm not sure it's quite right
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False, add_prefix_space=True)
        context_tokens = hackyEncode(tokenizer, hackyWhiteSpaceCutter(prompt)+context)
        context_tokens = context_tokens[-(maxHistory-len(prompt_tokens)):]
        #logger.debug('DECODED CONTEXT TOKENS: `%r`', tokenizer.convert_ids_to_tokens(context_tokens))
        prompt_tokens.extend(context_tokens)
        context_tokens = prompt_tokens
        #logger.debug('DECODED OUTPUT IS: `%r`', tokenizer.decode(context_tokens, clean_up_tokenization_spaces=False))
        #this is a hack and it should be up to the sampler to deal with max size
        if len(context_tokens) > maxHistory:
            logger.error("CONTEXT IS TOO LONG ERROR")
        return context_tokens

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

#premature optimization is the root of all evil
def sample_token(logits, genList, repetition_penalty, device):
    #many possible optimizations here
    #May help to switch to CPU at this stage
    #we don't need a softmax here but it's *possibly* more numerically stable than an exp.
    penalty_list={}
    for gtoken in genList:
        penalty_list[gtoken]=repetition_penalty
    probs = F.softmax(logits, dim=-1)
    selection_tokens=[]
    selection_logits=torch.zeros(NUM_SAMPLES, device=device)
    for i in range(NUM_SAMPLES):
        token = torch.multinomial(probs, num_samples=1)[0]
        tokenInt = token.item()
        selection_tokens.append(tokenInt)
        tokenStr = tokenizer.convert_ids_to_tokens(tokenInt)
        #print(tokenStr)
        #genList.append(token)
        if tokenStr in penalty_list:
            probs[tokenInt] /= penalty_list[tokenStr]
            selection_logits -= math.log(penalty_list[tokenStr])
            del penalty_list[tokenStr]

        #print(tokenizer.decode(genList))
        #genList.pop()

    token_index = torch.multinomial(F.softmax(selection_logits, dim=-1), num_samples=1).item()
    return selection_tokens[token_index]


#length should be max length, other settings should be removed, device should not be set
#we could possibly optimize this by having larger batch sizes but it would likely double or more the memory requirements
def sample_sequence(
    model,
    length,
    context,
    previousText=None,
    num_samples=1,
    temperature=1,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    too_short_penalty=5,
    min_gen_length=1,
    device="cpu",
    stop_tokens=None,
    tokenizer=None,
):
    #can probably remove this now
    logger.debug('temp: {}    top_k: {}    top_p: {}    rep-pen: {}'.format(temperature, top_k, top_p, repetition_penalty))
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    USE_PAST = True
    next_token = context
    outputs = None
    logger.warning("DELET THIS")
    top_k=1000
    top_p =0
    with torch.no_grad():
        for j in range(length):
            #why would we ever not use past?
            if USE_PAST:
                past = outputs[1] if outputs is not None else None
                inputs = {"input_ids": next_token, "past": past}
            else:
                inputs = {"input_ids": generated}

            outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)

            logits=outputs[0][:, -1, :][0].float()
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            logits = logits/(temperature if temperature > 0 else 1.0)

            shortPen=math.exp(too_short_penalty)
            if stop_tokens is not None:
                if j<min_gen_length:
                    for k in stop_tokens:
                        logits[stop_token]-=shortPen
                    
            genList = generated[0].tolist()
            if previousText is not None:
                genList.extend(previousText.tolist())
            expRepPen = math.exp(repetition_penalty)
            for k in set(genList):
                logits[k] -= expRepPen


            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            else:
                #token_id=sample_token(logits, genList, repetition_penalty, device)
                token_index = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
                next_token = torch.LongTensor([token_index]).to(device).unsqueeze(-1)#.unsqueeze(-1)

            generated = torch.cat((generated, next_token), dim=1)
            #disabled clean up of spaces, see what effect this has TODO
            genText = self.tokenizer.decode( o, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            if re.search(stop_pattern, genText):
                logger.debug('Stopping Generation Early as stop condition reached')
                break

    return generated


#def truncate_multiple_sequences(seqs, max_len=100):
#    """Truncate multiple sequences, longest first, removing first."""
#    while sum(len(s) for s in seqs) > max_len:
#        longest = sorted(seqs, key=len, reverse=True)[0]
#        longest.pop(0)

function strDtype(dtype):
    return re.search('\d+', str(dtype)).group(0)

#this class solves the problem of reusing different models
class ModelContainer:
    def __init__(model_path=Path('models', 'pytorch-16BIT-model_v5', device=None, dtype=None):
        if device is None:
            if settings.getboolean('force-cpu'):
                device='cpu'
            elif torch.cuda.is_available():
                device='cuda'
            else:
                logger.warning('CUDA is not available, you are limited to CPU only.')
                device='cpu'
        self.device=device
        self.dtype=dtype if dtype else (torch.float32 if device=='cpu' else torch.float16)
        self.checkpoint_path = Path(model_path)
        if os.environ.get("DEBUG_GPT2", False):
            self.checkpoint_path = Path('models', 'gpt2-small')
            logger.warning("using DEBUG_GPT2 MODE! This is just for devs to quickly check a small GPT2 model with poor output")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError("Could not find {} Make sure to download a pytorch model and put it in the models directory!".format(str(self.checkpoint_path)))

        logger.info('Device: {}    Cuda Available: {}    Force CPU: {}    Precision: {}    Model Path: {}'.format(
            self.device,
            torch.cuda.is_available(),
            settings.getboolean('force-cpu'),
            strDtype(self.dtype)+'-bit',
            str(self.checkpoint_path)))

        self.tokenizer = GPT2Tokenizer.from_pretrained(str(self.checkpoint_path))
        self.model = GPT2LMHeadModel.from_pretrained(str(self.checkpoint_path))
        if self.model.dtype != self.dtype:
            logger.warning("Model is {}-bits but you are running at {}-bits. It can be converted in memory fine. But you may benefit from a model that's natively {}-bits.".format(strDtype(self.model.dtype), strDtype(self.dtype), strDtype(self.model.dtype)))
        self.model.to(self.dtype).to(self.device)
        self.model.eval()



class GPT2Generator:
    def __init__(
        self, model_container=None, model_path=None, generate_num=60, stop_words=['<|endoftext|'], max_history_tokens=1024, temperature=0.4, top_k=0, top_p=0.9, dtype=None, device=None, repetition_penalty=1.05,
    ):
        self.generate_num = generate_num
        self.temp = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_history_tokens = max_history_tokens or (1024 - generate_num)
        self.stop_words = stop_words
        self.MC = model_container or ModelContainer(model_path, device=device, dtype=dtype)

    def sample_sequence(
        self, context_tokens=None, top_k=None, top_p=None, repetition_penalty=None, generate_num=None, temperature=None, stop_tokens=None
    ):
        generate_num = generate_num if (generate_num is not None) else self.generate_num
        temperature = temperature if (temperature is not None) else self.temp
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        out = sample_sequence(
            model=self.model,
            context=context_tokens,
            length=generate_num,
            # context=self.context,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_samples=self.samples,
            device=self.device,
            stop_tokens=stop_tokens,
            tokenizer=self.tokenizer
            # batch_size=self.batch_size,
        )
        return out

    #def prompt_replace(self, prompt):
        #if len(prompt) > 0 and prompt[-1] == " ":
        #    prompt = prompt[:-1]

        # prompt = second_to_first_person(prompt)
        return prompt

    def result_replace(self, result, allow_action=False):
        # logger.debug("BEFORE RESULT_REPLACE: `%s`", repr(result))

        result = cut_trailing_sentence(result, allow_action=allow_action)

        if len(result) == 0:
            return ""
        first_letter_capitalized = result[0].isupper()
        result = result.replace('."', '".')
        result = result.replace("#", "")
        result = result.replace("*", "")
        #TODO look at this I think blank lines should be fine or blacklisted at generation time
        result = result.replace("\n\n", "\n")
        # result = first_to_second_person(result)

        if not first_letter_capitalized:
            result = result[0].lower() + result[1:]

        #this is annoying since we can already see the AIs output
        #logger.debug( "AFTER RESULT_REPLACE: `%r`. allow_action=%r", repr(result), allow_action)

        return result

    def generate_raw(
            self, context, prompt='', generate_num=None, temperature=None, top_k=None, top_p=None, repetition_penalty=None, stop_tokens=None
    ):
        assert(top_k is not None)
        assert(temperature is not None)
        assert(top_p)
        assert(repetition_penalty)
            
        context_tokens=memory_merge(prompt, context, self.tokenizer, self.max_history_tokens)


        # if os.environ.get("DEBUG_GPT2", False):
        logger.debug(
            "Text passing into model `%r`",
            self.tokenizer.decode(
                context_tokens,
                clean_up_tokenization_spaces=True,
                #skip_special_tokens=True,
            ),
        ) 
        generated = 0
        for _ in range(self.samples // self.batch_size):
            out = self.sample_sequence(
                context_tokens,
                generate_num=generate_num,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_tokens=stop_tokens,
            )
            out = out[:, len(context_tokens) :].tolist()
            for o in out:
                generated += 1
                #disabled clean up of spaces, see what effect this has TODO
                text = self.tokenizer.decode( o, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                if self.stop_token:
                    index = text.find(self.stop_token)
                    if index == -1:
                        index = None
                    text = text[:index]
                if stop_tokens is not None:
                    for stop_token in stop_tokens:
                        index = text.find(self.stop_token)
                        if index == -1:
                            index = None
                        text = text[:index]
        return text

    def generate(self, context, prompt='', temperature=None, top_p=None, top_k=None, repetition_penalty=None, depth=0):
        assert(top_k is not None)
        assert(temperature is not None)
        assert(top_p)
        assert(repetition_penalty)
        #logger.debug("BEFORE PROMPT_REPLACE: `%r`", prompt)

        #prompt = [self.prompt_replace(p) for p in prompt]

        # logger.debug("AFTER PROMPT_REPLACE is: `%r`", repr(prompt))
        assert(prompt+context)

        text = self.generate_raw(
            context, prompt, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, stop_tokens=self.tokenizer.encode(["<|endoftext|>", ">"])
        )

        logger.debug("Generated result is: `%r`", repr(text))

        result = self.result_replace(text)

        if (depth > 6) and len(result) == 0:
            # Sometimes it keeps generating a story startng with an action (">"), if it's tried a few times and it keeps
            # happening, lets let it keep action text which starts in ">"
            # We could just blacklist that token and force it to generate something else. TODO
            result = self.result_replace(text, allow_action=True)
            logger.info(
                "Model generated empty text after formatting `%r`. Trying to format less with allow_action=True. `%r`",
                text,
                result,
            )

            #same here as above
        if len(result) == 0:
            if depth < 20:
                logger.info("Model generated empty text trying again %r", depth)
                return self.generate(
                    prompt, context, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, depth=depth + 1
                )
            else:
                logger.warn(
                    "Model generated empty text %r times. Try another action", depth
                )
        return result
