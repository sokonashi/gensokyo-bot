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

#Should remove trailing spaces but make sure to test this #TODO
#TODO in fact we can completely fill the AI's memory instead of cutting the first 60 tokens, if we are willing to push the cut context to the front
def memory_merge(prompt, context, tokenizer, maxHistory=1024):
    assert(prompt+context)
    #logger.debug('RAW TEXT INPUT IS:`%r`', context)
    #I'm going with the add prefix option but I'm not sure it's optimal
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False, add_prefix_space=True)
    context_tokens = hackyEncode(tokenizer, hackyWhiteSpaceCutter(prompt)+context)
    if len(prompt_tokens) > maxHistory/2.0:
        logger.warning("More than half of memory is prompt.")
    if len(prompt_tokens) > maxHistory:
        logger.error("The prompt is larger than the memory limit.")
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
        #genList.append(token)
        if tokenStr in penalty_list:
            probs[tokenInt] /= penalty_list[tokenStr]
            selection_logits -= math.log(penalty_list[tokenStr])
            del penalty_list[tokenStr]

        #print(tokenizer.decode(genList))
        #genList.pop()

    token_index = torch.multinomial(F.softmax(selection_logits, dim=-1), num_samples=1).item()
    return selection_tokens[token_index]



#def truncate_multiple_sequences(seqs, max_len=100):
#    """Truncate multiple sequences, longest first, removing first."""
#    while sum(len(s) for s in seqs) > max_len:
#        longest = sorted(seqs, key=len, reverse=True)[0]
#        longest.pop(0)

def strDtype(dtype):
    return re.search('\d+', str(dtype)).group(0)

#this class solves the problem of reusing the same model with different generators
class ModelContainer:
    def __init__(self, model_path=Path('models', 'pytorch-16BIT-model_v5'), device=None, dtype=None):
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
        modelDtype=next(self.model.parameters()).dtype
        if modelDtype != self.dtype:
            logger.warning("Model is {}-bits but you are running at {}-bits. It can be converted in memory fine. But you may benefit from a model that's natively {}-bits.".format(strDtype(modelDtype), strDtype(self.dtype), strDtype(self.dtype)))
        self.model.to(self.dtype).to(self.device)
        self.model.eval()



class GPT2Generator:
    def __init__(
        self, model_container=None, model_path=None, generate_num=60, stop_patterns=[r'\<\|endoftext\|\>'], max_history_tokens=None, temperature=0.4, top_k=0, top_p=0.9, numBeams=2, wordPenalties=None, dtype=None, device=None, repetition_penalty=1.05,
    ):
        self.generate_num = generate_num
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_history_tokens = max_history_tokens or (1024 - generate_num)
        self.numBeams = numBeams
        self.stop_patterns = stop_patterns
        self.MC = model_container or ModelContainer(model_path, device=device, dtype=dtype)
        self.wordPenalties=None
        if wordPenalties is not None:
            setWordPenalties(wordPenalties)

    def setWordPenalties(self, wordPenalties):
        self.wordPenalties = torch.zeros(len(self.MC.tokenizer.encoder), device=self.MC.device)
        for regex in wordPenalties:
            for k in self.MC.tokenizer.encoder:
                if re.search(regex, k, re.IGNORECASE):
                    weight = float(wordPenalties[regex])/math.log2(math.e)
                    self.wordPenalties[self.MC.tokenizer.encoder[k]]=weight
                    #disable this as it will be annoying
                    logger.info('Token {} matched {}, giving weight e^{}'.format(repr(k), repr(regex), weight))

    def beamSearch(self, context, numBeams, beamDepth, previousText=[]):
        while true:
            genTexts = []
            logProbs = []
            for i in range(numBeams):
                genText[i], logProb=self.sample(context, previousText)


    def sample(
        self,
        context,
        length,
        previousText=None
    ):
        context = torch.tensor(context, dtype=torch.long, device=self.MC.device)
        context = context.unsqueeze(0)
        #generated = context
        next_token = context
        outputs = None
        genTokens = []
        logProb = 0
        #with torch.no_grad():
        #    print(self.MC.model(input_ids=next_token, past=None))
        with torch.no_grad():
            for j in range(length):
                outputs = self.MC.model(input_ids=next_token, past=outputs[1] if outputs is not None else None)
                logits=outputs[0][:, -1, :]
                origLogits = logits[0].float().clone().detach()
                logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
                logits = logits[0].float()
    
                logits = logits/(self.temperature if self.temperature > 0 else 1.0)

                if self.wordPenalties is not None:
                    logits-=self.wordPenalties
    
                #genList = generated[0].tolist()
                genList = genTokens.copy()
                if previousText is not None:
                    genList.extend(previousText.tolist())
                expRepPen = math.exp(self.repetition_penalty)
                for k in set(genList):
                    logits[k] -= expRepPen
    
                if self.temperature == 0:  # greedy sampling:
                    token_index = torch.argmax(logits, dim=-1).item()
                else:
                    #token_id=sample_token(logits, genList, self.repetition_penalty, self.MC.device)
                    token_index = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
                
                next_token = torch.LongTensor([token_index]).to(self.MC.device).unsqueeze(-1)#.unsqueeze(-1)
    
                logProb += origLogits[token_index]
                genTokens.append(token_index)
                #disabled clean up of spaces, see what effect this has TODO
                genText = self.MC.tokenizer.decode(genTokens, clean_up_tokenization_spaces=False, skip_special_tokens=False)
                stopped = False
                for p in self.stop_patterns:
                    if re.search(p, genText):
                        logger.debug('Stopping Generation Early as stop condition reached')
                        stopped = True
                        break
                if stopped:
                    break

        return genTokens, logProb

    #TODO test reenable some of this
    # The details of what needs to be cleaned up are model and training data dependent
    #def result_replace(self, result, allow_action=False):
        # logger.debug("BEFORE RESULT_REPLACE: `%s`", repr(result))

        #result = cut_trailing_sentence(result, allow_action=allow_action)

        #if len(result) == 0:
        #    return ""
        #first_letter_capitalized = result[0].isupper()
        #result = result.replace('."', '".')
        #result = result.replace("#", "")
        #result = result.replace("*", "")
        #result = result.replace("\n\n", "\n")
        # result = first_to_second_person(result)

        #if not first_letter_capitalized:
        #    result = result[0].lower() + result[1:]

        #this is annoying since we can already see the AIs output
        #logger.debug( "AFTER RESULT_REPLACE: `%r`. allow_action=%r", repr(result), allow_action)

        #return result

            
    def generate(self, context, prompt=''):
        context_tokens=memory_merge(prompt, context, self.MC.tokenizer, self.max_history_tokens)
        logger.debug( "Text passing into model `%r`", self.MC.tokenizer.decode(context_tokens))
        out, logP = self.sample(context_tokens, self.generate_num)
        #disabled clean up of spaces, see what effect this has TODO
        logger.debug( "Generated Result: `%r`", self.MC.tokenizer.decode(out))
        return self.MC.tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=False)
