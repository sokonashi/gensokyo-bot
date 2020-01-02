from pathlib import Path
#remove this in a few days
with open(Path('interface', 'start-message.txt'), 'r') as file:
    print('\x1B[7m'+file.read()+'\x1B[27m')
import gc
import random
import torch
import textwrap
import sys
from random import shuffle
from shutil import get_terminal_size

from getconfig import config, settings, colors, logger
#from story.story_manager import *
from storymanager import Story
from utils import *
from gpt2generator import GPT2Generator
from interface import instructions


# TODO: Move all these utilty functions to seperate utily file

# add color for windows users that install colorama
#   It is not necessary to install colorama on most systems
try:
    import colorama
    colorama.init()
except ModuleNotFoundError:
    pass

#any documentation on what codes are supported?
def _in_colab():
    """Some terminal codes don't work in a colab notebook."""
    # from https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
    try:
        from IPython import get_ipython
        if (not get_ipython()) or ('IPKernelApp' not in get_ipython().config):  # pragma: no cover
            raise ImportError("console")
        if 'VSCODE_PID' in os.environ:  # pragma: no cover
            raise ImportError("vscode")
    except ImportError:
        if get_terminal_size()[0]==0 or 'google.colab' in sys.modules:
            return True
        return False
    else:
        return True

IN_COLAB = _in_colab()
logger.info("Colab detected: {}".format(IN_COLAB))
IN_COLAB = IN_COLAB or settings.getboolean('colab-mode') 
if IN_COLAB:
    logger.warning("Colab mode enabled, disabling line clearing and readline to avoid colab bugs.")
else:
    try:
        import readline
        logger.info('readline has been imported. This enables a number of editting features but may cause bugs for colab users.')
    except ModuleNotFoundError:
        pass



termWidth = get_terminal_size()[0]
if termWidth < 5:
    logger.warning("Your detected terminal width is: "+str(get_terminal_size()[0]))
    termWidth = 999999999

# ECMA-48 set graphics codes for the curious. Check out "man console_codes"
def colPrint(text, col="0", wrap=True, end=None):
    if wrap:
        width = settings.getint("text-wrap-width")
        width = 999999999 if width < 2 else width
        width=min(width, termWidth)
        text = textwrap.fill(
            text, width, replace_whitespace=False
        )
    print("\x1B[{}m{}\x1B[{}m".format(col, text, colors["default"]), end=end)
    return text.count('\n')+1


def colInput(str, col1=colors["default"], col2=colors["default"]):
    val = input("\x1B[{}m{}\x1B[0m\x1B[{}m".format(col1, str, col1))
    print("\x1B[0m", end="")
    return val


def clear_lines(n):
    """Clear the last line in the terminal."""
    if IN_COLAB:
        # this wont work in colab etc
        return
    screen_code = "\033[1A[\033[2K"  # up one line, and clear line
    for _ in range(n):
        print(screen_code, end="")


def getNumberInput(n):
    bell()
    val = colInput(
        "Enter a number from above (default 0):",
        colors["selection-prompt"],
        colors["selection-value"],
    )
    if val == "":
        return 0
    elif not re.match("^\d+$", val) or 0 > int(val) or int(val) > n:
        colPrint("Invalid choice.", colors["error"])
        return getNumberInput(n)
    else:
        return int(val)


def selectFile(p=Path("prompts")):
    if p.is_dir():
        files = [x for x in p.iterdir()]
        #TODO: make this a config option (although really it should be random)
        shuffle(files)
        for n in range(len(files)):
            colPrint(
                "{}: {}".format(n, re.sub(r"\.txt$", "", files[n].name)), colors["menu"]
            )
        return selectFile(files[getNumberInput(len(files) - 1)])
    else:
        with p.open("r", encoding="utf-8") as file:
            line1 = file.readline()
            rest = file.read()
        return (line1, rest)


def getGenerator():
    colPrint(
        "\nInitializing AI Engine! (This might take a few minutes)\n",
        colors["loading-message"],
    )
    models=[x for x in Path('models').iterdir() if x.is_dir()]
    if not models:
        raise FileNotFoundError('There are no models in the models directory! You must download a pytorch compatible model!')
    elif len(models) >1:
        colPrint("You have multiple models in your models folder. Please select one to load:", colors['message']) 
        for n, model_path in enumerate(models):
            colPrint("{}: {}".format(n, model_path.name), colors['menu'])
        
        model=models[getNumberInput(len(models)-1)]
    else:
        model=models[0]
        logger.info("Using model: "+str(model))
    return GPT2Generator(
        model_path=model,
        generate_num=settings.getint("generate-num"),
        temperature=settings.getfloat("temp"),
        top_k=settings.getint("top-keks"),
        top_p=settings.getfloat("top-p"),
        repetition_penalty=settings.getfloat("rep-pen"),
    )


if not Path("prompts", "Anime").exists():
    try:
        import pastebin
    except:
        #temporary fix "e is not defined"
        #logger.warning("Failed to scrape pastebin: %e", e)
        colPrint(
            "Failed to scrape pastebin, possible connection issue.\nTry again later. Continuing without downloading prompts...",
            colors["error"],
        )


#class AIPlayer:
    #def __init__(self, story):
        #self.story = story

    #def get_action(self):
        # While we want the story to be on track, but not to on track that it loops
        # the actions can be quite random, and this helps inject some user curated randomness
        # and prevent loops. So lets make the actions quite random, and prevent duplicates while we are at it

        # what to feed to model?
        #mem_ind = random.randint(1, 6) # How many steps to include
        #sample = random.randint(0, 1) # Random steps from history?
        #include_prompt = random.randint(0, 1) # Include the initial promts
        #predicates = ['You try to ', 'You say "', 'You start to ', '"']  # The model has to continue from here
        
        #predicate = random.sample(predicates, 1)[0]
        #action_prompt = self.story_manager.story_context(
        #    mem_ind,
        #    sample,
        #    include_prompt
        #)
        #action_prompt[-1] = action_prompt[-1].strip() + "\n> "+predicate
        #print(action_prompt)

        #result_raw = self.story_manager.generator.generate(
        #    action_prompt,
        #    self.story_manager.prompt,
        #    generate_num=settings.getint("action-generate-num"),
        #    temperature=settings.getfloat("action-temp"),
        #    stop_tokens=self.story_manager.generator.tokenizer.encode(["<|endoftext|>", "\n", ">"])
        #    # stop_tokens=self.generator.tokenizer.encode(['>', '<|endoftext|>'])
        #)
        #logger.debug("get_action (mem_ind=%s, sample=%s, include_prompt=%s, predicate=`%r`) -> %r", mem_ind, sample, include_prompt, predicate, result_raw)
        #result = predicate + result_raw.lstrip()
        #result = clean_suggested_action(
        #    result, min_length=settings.getint("action-min-length")
        #)
        # Sometimes the suggestion start with "You" we will add that on later anyway so remove it here
        #result = re.sub("^ ?[Yy]ou try to ?", "You ", result)
        #result = re.sub("^ ?[Yy]ou start to ?", "You ", result)
        #result = re.sub("^ ?[Yy]ou say \"", "\"", result)
        #result = re.sub("^ ?[Yy]ou ?", "", result)
        #return result
        #return self.story.getSuggestion()


def bell():
    if settings.getboolean("console-bell"):
        print("\x07", end="")

def d20ify_speech(action, d):
    adjectives_say_d01 = [
        "mumble",
        "prattle",
        "incoherently say",
        "whine",
        "ramble",
        "wheeze",
    ]
    adjectives_say_d20 = [
        "successfully",
        "persuasively",
        "expertly",
        "conclusively",
        "dramatically",
        "adroitly",
        "aptly",
    ]
    if d == 1:
        adjective = random.sample(adjectives_say_d01, 1)[0]
        action = "You " + adjective + " " + action
    elif d == 20:
        adjective = random.sample(adjectives_say_d20, 1)[0]
        action = "You " + adjective + " say " + action
    else:
        action = "You say " + action
    return action

def d20ify_action(action, d):
    adjective_action_d01 = [
        "disastrously",
        "incompetently",
        "dangerously",
        "stupidly",
        "horribly",
        "miserably",
        "sadly",
    ]
    adjective_action_d20 = [
        "successfully",
        "expertly",
        "conclusively",
        "adroitly",
        "aptly",
        "masterfully",
    ]
    if d == 1:
        adjective = random.sample(adjective_action_d01, 1)[
            0
        ]
        action = "You " + adjective + " fail to " + action
    elif d < 5:
        action = "You attempt to " + action
    elif d < 10:
        action = "You try to " + action
    elif d < 15:
        action = "You start to " + action
    elif d < 20:
        action = "You " + action
    else:
        adjective = random.sample(adjective_action_d20, 1)[
            0
        ]
        action = "You " + adjective + " " + action
    return action

def newStory(generator, prompt, context):
    story = Story(generator, prompt)
    assert (story.prompt)
    first_result = story.act(context)
    colPrint(prompt, colors['user-text'], end='')
    colPrint(context, colors['user-text'], end='')
    colPrint(first_result[0], colors['ai-text'], end='')
    print("\n\n")
    return story

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|ca|gg|tv|co|net|org|io|gov)"

def splitIntoSentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace(".<stop>\"", ".\"<stop>")
    text = text.replace("?<stop>\"", "?\"<stop>")
    text = text.replace("!<stop>\"", "!\"<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def listSentences(sentences):
    i = 0
    for s in sentences:
        colPrint(str(i) + ") " + s, colors['menu'])
        i += 1
    colPrint(str(len(sentences)) + ") (Back)", colors['menu'])

def alterText(text):
    sentences = splitIntoSentences(text)
    while True:
        colPrint("\n" + " ".join(sentences) + "\n", colors['menu'])
        colPrint("\n0) Edit a sentence.\n1) Remove a sentence.\n2) Add a sentence.\n3) Edit entire prompt.\n4) Save and finish.", colors['menu'], wrap=False)
        colPrint("\nChoose an option: ", colors['menu'], wrap=False)
        try:
            i = getNumberInput(4)
        except:
            continue
        if i == 0:
            while True:
                listSentences(sentences)
                i = getNumberInput(len(sentences))
                if i == len(sentences):
                    break
                else:
                    colPrint("\n" + sentences[i], colors['menu'])
                    sentences[i] = colInput("\nEnter the altered sentence: ", colors['menu'])
        elif i == 1:
            while True:
                listSentences(sentences)
                i = getNumberInput(len(sentences))
                if i == len(sentences):
                    break
                else:
                    del sentences[i]
        elif i == 2:
            while True:
                listSentences(sentences)
                i = getNumberInput(len(sentences))
                if i == len(sentences):
                    break
                else:
                    sentences.insert(i+1, colInput("\nEnter the new sentence: ", colors['menu']))
        elif i == 3:
            colPrint("\n" + " ".join(sentences), colors['menu'])
            text = colInput("\nEnter the new altered prompt: ", colors['menu'])
            sentences = splitIntoSentences(text)
        elif i == 4:
            break
    return " ".join(sentences)

def play(generator):
    print("\n")

    with open(Path("interface", "mainTitle.txt"), "r", encoding="utf-8") as file:
        colPrint(file.read(), colors["title"], wrap=False)

    with open(Path("interface", "subTitle.txt"), "r", encoding="utf-8") as file:
        cols = termWidth
        for line in file:
            line=re.sub(r'\n', '', line)
            line=line[:cols]
            #fills in the graphic using reverse video mode substituted into the areas between |'s
            colPrint(re.sub(r'\|[ _]*(\||$)', lambda x: '\x1B[7m'+x.group(0)+'\x1B[27m', line), colors['subtitle'], False)

    print()
    colPrint("Go to https://github.com/cloveranon/Clover-Edition/ or email cloveranon@nuke.africa for bug reports, help, and feature requests.", colors['subsubtitle'])

    while True:
        # May be needed to avoid out of mem
        gc.collect()
        torch.cuda.empty_cache()

        print("\n\n")

        colPrint("0: Pick Prompt From File (Default if you type nothing)\n1: Write Custom Prompt", colors['menu'])

        if getNumberInput(1) == 1:
            with open(
                Path("interface", "prompt-instructions.txt"), "r", encoding="utf-8"
            ) as file:
                colPrint(file.read(), colors["instructions"], False)
            prompt = colInput("Prompt>", colors["main-prompt"], colors["user-text"])
            context = colInput("Context>", colors["main-prompt"], colors["user-text"])
            filename = colInput(
                "Name to save prompt as? (Leave blank for no save): ",
                colors["query"],
                colors["user-text"],
            )
            filename = re.sub(
                "-$", "", re.sub("^-", "", re.sub("[^a-zA-Z0-9_-]+", "-", filename))
            )
            if filename != "":
                with open(
                    Path("prompts", filename + ".txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(context + "\n" + prompt)
        else:
            prompt, context = selectFile()
        assert(prompt+context)

        instructions()

        print()
        colPrint("Generating story...", colors["loading-message"])

        story = newStory(generator, prompt, context)

        while True:
            # Generate suggested actions
            act_alts = settings.getint("action-sugg")
            if act_alts > 0:

                # TODO change this to two messages for different colors
                suggested_actions = []
                colPrint("\nSuggested actions:", colors["selection-value"])
                action_suggestion_lines = 2
                for i in range(act_alts):
                    suggested_action = story.getSuggestion()
                    if len(suggested_action.strip()) > 0:
                        j = len(suggested_actions)
                        suggested_actions.append(suggested_action)
                        suggestion = "{}> {}".format(j, suggested_action)
                        action_suggestion_lines += colPrint(suggestion, colors["selection-value"])
                print()

            bell()
            action = colInput("> You ", colors["main-prompt"], colors["user-text"])

            # Clear suggestions and user input
            if act_alts > 0:
                action_suggestion_lines += 2
                if not IN_COLAB:
                    clear_lines(action_suggestion_lines)

                    # Show user input again
                    # colPrint("\n> " + action.rstrip(), colors["user-text"], end="")

            cmdRegex = re.search("^/([^ ]+) *(.*)$", action)

            # If this is a command
            if cmdRegex:
                action = cmdRegex.group(1)
                cmdArgs = cmdRegex.group(2).strip().split()
                if action == "set":
                    if len(cmdArgs) < 2:
                        colPrint("Invalid number of arguments for set command.\n", colors["error"])
                        instructions()
                        continue
                    if cmdArgs[0] in settings:
                        currentSettingValue = settings[cmdArgs[0]]
                        colPrint(
                            "Current Value of {}: {}     Changing to: {}".format(
                                cmdArgs[0], currentSettingValue, cmdArgs[1]
                            )
                        )
                        settings[cmdArgs[0]] = cmdArgs[1]
                        colPrint("Save config file?", colors["query"])
                        colPrint(
                            "Saving an invalid option will corrupt file!", colors["error"]
                        )
                        if (
                            colInput(
                                "y/n? >",
                                colors["selection-prompt"],
                                colors["selection-value"],
                            )
                            == "y"
                        ):
                            with open("config.ini", "w", encoding="utf-8") as file:
                                config.write(file)
                    else:
                        colPrint("Invalid setting", colors["error"])
                        instructions()

                elif action == "menu":
                    break

                elif action == "restart":
                    print()
                    colPrint("Restarting story...", colors["loading-message"])
                    story = newStory(generator, story.prompt, context)
                    continue

                elif action == "quit":
                    exit()

                elif action == "help":
                    instructions()

                elif action == "print":
                    print("\nPRINTING\n")
                    #TODO colorize printed story
                    colPrint(str(story), colors["print-story"])

                elif action == "retry":
                    if len(story.story) == 1:
                        print()
                        colPrint("Restarting story...", colors["loading-message"])
                        story = newStory(generator, story.prompt, context)
                        continue
                    else:
                        newaction = story.story[-1][0]
                    colPrint(newaction, colors['user-text'], end='')
                    story.story=story.story[:-1]
                    result = "\n" + story.act(newaction)[0]
                    if len(story.story) >= 2:
                        similarity = get_similarity(result, story.story[-2][1][0])
                        if similarity > 0.9:
                            story.story = story.story[:-1]
                            colPrint(
                                "Woops that action caused the model to start looping. Try a different action to prevent that.",
                                colors["error"],
                            )
                            continue
                    colPrint(result, colors["ai-text"])
                    continue

                elif action == "revert":
                    if len(story.story) == 1:
                        colPrint("You can't go back any farther. ", colors["error"])
                        continue
                    story.story=story.story[:-1]
                    colPrint("Last action reverted. ", colors["message"])
                    if len(story.story)<2:
                        colPrint(story.prompt, colors["ai-text"])
                    colPrint(story.story[-1][1][0], colors["ai-text"])
                    continue

                elif action == "alter":
                    story.story[-1][1][0] = alterText(story.story[-1][1][0])
                    if len(story.story)<2:
                        colPrint(story.prompt, colors["ai-text"])
                    else:
                        colPrint("\n" + story.story[-1][0] + "\n", colors["transformed-user-text"])
                    colPrint("\n" + story.story[-1][1][0] + "\n\n", colors["ai-text"])

                elif action == "prompt":
                    story.prompt = alterText(story.prompt)
                    if len(story.story)<2:
                        colPrint(story.prompt, colors["ai-text"])
                    else:
                        colPrint("\n" + story.story[-1][0] + "\n", colors["transformed-user-text"])
                    colPrint("\n" + story.story[-1][1][0] + "\n\n", colors["ai-text"])

                else:
                    colPrint("Invalid command: " + action, colors["error"])

            # Otherwise this is just a normal action.
            else:
                if act_alts > 0:
                    # Options to select a suggestion action
                    if action in [str(i) for i in range(len(suggested_actions))]:
                        action = suggested_actions[int(action)]

                original_action=action
                action = action.strip()
                #TODO debug stuff to delete
                if action != original_action:
                    logger.debug("STRIPPED WHITE SPACE OFF ACTION %r vs %r", original_action, action)

                # Crop actions to a max length
                #action = action[:4096]

                if action != "":

                    # Roll a 20 sided dice to make things interesting
                    d = random.randint(1, 20)
                    logger.debug("roll d20=%s", d)

                    # If it says 'You say "' then it's still dialouge. Normalise it by removing `You say `, we will add again soon
                    action = re.sub("^ ?[Yy]ou say [\"']", '"', action)
                    if any(action.lstrip().startswith(t) for t in ['"', "'"]):
                        if settings.getboolean("action-d20"):
                            action = d20ify_speech(action, d)
                        else:
                            action = "You say " + action
                        logger.info("%r. %r, %r", action, any(action.lstrip().startswith(t) for t in ['"', "'"]), settings.getboolean("action-d20"))
                    else:
                        action = first_to_second_person(action)
                        if not action.lower().startswith(
                            "you "
                        ) and not action.lower().startswith("i "):
                            action = action[0].lower() + action[1:]
                            # roll a d20
                            if settings.getboolean("action-d20"):
                                action = d20ify_action(action, d)
                            else:
                                action = "You " + action

                        if action[-1] not in [".", "?", "!"]:
                            action = action + "."

                action = "\n> " + action + "\n"

                colPrint(
                    "\n>" + action.lstrip().lstrip("> \n"),
                    colors["transformed-user-text"],
                )
                #TODO check if leading white space makes sense
                result = "\n" + story.act(action)[0]

                #TODO: Replace all this nonsense
                if len(story.story) >= 2:
                    similarity = get_similarity(result, story.story[-2][1][0])
                    if similarity > 0.9:
                        story.story = story.story[:-1]
                        colPrint(
                            "Woops that action caused the model to start looping. Try a different action to prevent that.",
                            colors["error"],
                        )
                        continue

                if player_won(result):
                    colPrint(result + "\n CONGRATS YOU WIN", colors["message"])
                    break
                elif player_died(result):
                    colPrint(result, colors["ai-text"])
                    colPrint("YOU DIED. GAME OVER", colors["error"])
                    colPrint(
                        "\nOptions:\n0)Start a new game\n1)\"I'm not dead yet!\" (If you didn't actually die)",
                        colors["menu"],
                    )
                    choice = getNumberInput(1)
                    if choice == 0:
                        break
                    else:
                        colPrint("Sorry about that...where were we?", colors["query"])
                colPrint(result, colors["ai-text"])


# This is here for rapid development, without reloading the model. You import play into a jupyternotebook with autoreload
if __name__ == "__main__":
    with open(Path("interface", "clover"), "r", encoding="utf-8") as file:
        print(file.read())
    generator = getGenerator()
    play(generator)
