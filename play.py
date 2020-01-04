from pathlib import Path

# remove this in a few days
with open(Path('interface', 'start-message.txt'), 'r') as file:
    print('\x1B[7m' + file.read() + '\x1B[27m')
import gc
import torch
import random
from random import shuffle

from getconfig import config
from storymanager import Story
from utils import *
from gpt2generator import GPT2Generator
from interface import instructions

# add color for windows users that install colorama
#   It is not necessary to install colorama on most systems
try:
    import colorama

    colorama.init()
except ModuleNotFoundError:
    pass

logger.info("Colab detected: {}".format(in_colab()))

def select_file(p=Path("prompts")):
    if p.is_dir():
        files = [x for x in p.iterdir()]
        # TODO: make this a config option (although really it should be random)
        shuffle(files)
        list_items([f.name for f in files], colors["menu"])
        return select_file(files[input_number(len(files) - 1)])
    else:
        with p.open("r", encoding="utf-8") as file:
            try:
                lines = file.read().strip().split('\n')
                if len(lines) < 2:
                    context = ""
                    prompt = lines[0]
                else:
                    context = lines[0]
                    prompt = lines[1]
            except IOError:
                return None, None
        return context, prompt


def get_generator():
    output(
        "\nInitializing AI Engine! (This might take a few minutes)",
        colors["loading-message"], end="\n\n"
    )
    models = [x for x in Path('models').iterdir() if x.is_dir()]
    if not models:
        raise FileNotFoundError(
            'There are no models in the models directory! You must download a pytorch compatible model!')
    elif len(models) > 1:
        output("You have multiple models in your models folder. Please select one to load:", colors['message'])
        for n, model_path in enumerate(models):
            output("{}: {}".format(n, model_path.name), colors['menu'])

        model = models[input_number(len(models) - 1)]
    else:
        model = models[0]
        logger.info("Using model: " + str(model))
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
        output("Continuing without downloading prompts...",colors["error"],)



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


def new_story(generator, prompt, context):
    prompt = prompt.strip()
    context = context.strip()
    story = Story(generator, prompt)
    story.act(context)
    story.print_story()
    return story


def save_story(story):
    """Saves the existing story to a json file in the saves directory to be resumed later."""
    savefile = story.savefile
    while True:
        print()
        temp_savefile = input_line("Please enter a name for this save: ",
                                   colors["query"], colors["user-text"])
        savefile = temp_savefile if len(temp_savefile.strip()) > 0 else savefile
        if len(savefile.strip()) == 0:
            output("Please enter a valid savefile name. ", colors["error"])
        else:
            break
    savefile = os.path.splitext(savefile.lstrip("saves/").strip())[0]
    story.savefile = savefile
    savedata = story.to_json()
    Path("saves/").mkdir(parents=True, exist_ok=True)
    with open("saves/" + savefile + ".json", 'w') as f:
        try:
            f.write(savedata)
            output("Successfully saved to " + savefile, colors["message"])
        except IOError:
            output("Unable to write to file; aborting. ", colors["error"])


def load_story():
    """Prompts the user for a save file stored in the saves directory,
    and returns a valid story, as well as the prompt and starting context if the story is successfully loaded.
    Otherwise, returns None, None, None"""
    savefile = input_line("Enter the save you want to load: ",
                          colors["query"], colors["user-text"])
    if savefile.strip() == "":
        output("Invalid savename. ", colors["error"])
        return None, None, None
    else:
        try:
            savefile = os.path.splitext(savefile.lstrip("saves/").strip())[0]
            f = open("saves/" + savefile + ".json", 'r')
            story = Story(generator, "")
            story.savefile = savefile
            story.from_json(f.read())
            return story, story.context, story.actions[-1] if len(story.actions) > 0 else ""
        except FileNotFoundError:
            output("Save file not found. ", colors["error"])
            return None, None, None
        except IOError:
            output("Something wrong occurred when attempting to load the file. ", colors["error"])
            return None, None, None


def alter_text(text):
    sentences = sentence_split(text)
    while True:
        output(" ".join(sentences), colors['menu'])
        list_items(
            [
                "Edit a sentence.",
                "Remove a sentence.",
                "Add a sentence.",
                "Edit entire prompt.",
                "Save and finish."
            ], colors['menu'])
        try:
            i = input_number(4)
        except:
            continue
        if i == 0:
            while True:
                output("Choose the sentence you want to edit.", colors["menu"])
                list_items(sentences + ["(Back)"], colors["menu"])
                i = input_number(len(sentences))
                if i == len(sentences):
                    break
                else:
                    output(sentences[i], colors['menu'])
                    res = input_line("Enter the altered sentence: ", colors['menu']).strip()
                    if len(res) == 0:
                        output("Invalid sentence entered: returning to previous menu. ", colors['error'])
                        continue
                    sentences[i] = res
        elif i == 1:
            while True:
                output("Choose the sentence you want to remove.", colors["menu"])
                list_items(sentences + ["(Back)"], colors["menu"])
                i = input_number(len(sentences))
                if i == len(sentences):
                    break
                else:
                    del sentences[i]
        elif i == 2:
            while True:
                output("Choose the sentence you want to insert after.", colors["menu"])
                list_items(["(Beginning)"] + sentences + ["(Back)"], colors["menu"])
                max = len(sentences) + 1
                i = input_number(max)
                if i == max:
                    break
                else:
                    res = input_line("Enter the new sentence: ", colors['menu']).strip()
                    if len(res) == 0:
                        output("Invalid sentence entered: returning to previous menu. ", colors['error'])
                        continue
                    sentences.insert(i, res)
        elif i == 3:
            output(" ".join(sentences), colors['menu'])
            res = input_line("Enter the new altered prompt: ", colors['menu']).strip()
            if len(res) == 0:
                output("Invalid prompt entered: returning to previous menu. ", colors['error'])
                continue
            text = res
            sentences = sentence_split(res)
        elif i == 4:
            break
    return " ".join(sentences).strip()


def play(generator):
    print()

    with open(Path("interface", "mainTitle.txt"), "r", encoding="utf-8") as file:
        output(file.read(), colors["title"], wrap=False, beg='')

    with open(Path("interface", "subTitle.txt"), "r", encoding="utf-8") as file:
        cols = termWidth
        for line in file:
            line=re.sub(r'\n', '', line)
            line=line[:cols]
            #fills in the graphic using reverse video mode substituted into the areas between |'s
            if use_ptoolkit():
                style = Style.from_dict({
                    'nor': colors['subtitle'],
                    'rev': colors['subtitle'] + ' reverse',
                })
                text = re.sub(r'\|[ _]*(\||$)', lambda x: '<rev>'+x.group(0)+'</rev>', line)
                print_formatted_text(HTML('<nor>' + text + '</nor>'), style=style)
            else:
                output(re.sub(r'\|[ _]*(\||$)', lambda x: '\x1B[7m' + x.group(0) + '\x1B[27m', line), colors['subtitle'],
                       wrap=False, beg='')

    output("Go to https://github.com/cloveranon/Clover-Edition/ "
           "or email cloveranon@nuke.africa for bug reports, help, and feature requests.",
           colors['subsubtitle'])

    # Prevent reference before assignment
    story = None
    context = None
    prompt = None

    while True:
        # May be needed to avoid out of mem
        gc.collect()
        torch.cuda.empty_cache()

        list_items(["Pick Prompt From File (Default if you type nothing)", "Write Custom Prompt", "Load a Saved Game"],
                   colors['menu'])
        new_game_option = input_number(2)

        if new_game_option == 0:
            context, prompt = select_file()
            if context is None and prompt is None:
                output("Invalid prompt and context. Please try another file. ", colors["error"])
                continue
        elif new_game_option == 1:
            with open(
                    Path("interface", "prompt-instructions.txt"), "r", encoding="utf-8"
            ) as file:
                output(file.read(), colors["instructions"], wrap=False)
            if use_ptoolkit():
                output("Prompt>", colors["main-prompt"])
                prompt = edit_multiline()
                output("Context>", colors["main-prompt"])
                context = edit_multiline()
            else:
                context = input_line("Context> ", colors["main-prompt"], colors["user-text"])
                prompt = input_line("Prompt> ", colors["main-prompt"], colors["user-text"])
            filename = input_line(
                "Name to save prompt as? (Leave blank for no save): ",
                colors["query"],
                colors["user-text"],
            )
            filename = re.sub(
                "-$", "", re.sub("^-", "", re.sub("[^a-zA-Z0-9_-]+", "-", filename))
            )
            if filename != "":
                try:
                    with open(
                        Path("prompts", filename + ".txt"), "w", encoding="utf-8"
                    ) as f:
                        f.write(context + "\n" + prompt)
                except IOError:
                    output("Permission error! Unable to save custom prompt. ", colors["error"])
        elif new_game_option == 2:
            story, context, prompt = load_story()
            if not story:
                continue

        if len((context + prompt).strip()) == 0:
            output("Story has no prompt or context. Please enter a valid custom prompt. ", colors["error"])
            continue

        instructions()

        if story is None:
            output("Generating story...", colors["loading-message"])
            story = new_story(generator, context, prompt)
        else:
            output("Loading story...", colors["loading-message"])
            story.print_story()

        while True:
            # Generate suggested actions
            act_alts = settings.getint("action-sugg")
            if act_alts > 0:
                # TODO change this to two messages for different colors
                suggested_actions = []
                output("\nSuggested actions:", colors["selection-value"])
                action_suggestion_lines = 2
                for i in range(act_alts):
                    suggested_action = story.get_suggestion()
                    if len(suggested_action.strip()) > 0:
                        j = len(suggested_actions)
                        suggested_actions.append(suggested_action)
                        suggestion = "{}> {}".format(j, suggested_action)
                        action_suggestion_lines += output(suggestion, colors["selection-value"])
                print()

            bell()
            print()
            action = input_line("> You ", colors["main-prompt"], colors["user-text"])

            # Clear suggestions and user input
            if act_alts > 0:
                action_suggestion_lines += 2
                clear_lines(action_suggestion_lines)

            cmdRegex = re.search("^/([^ ]+) *(.*)$", action)

            # If this is a command
            if cmdRegex:
                action = cmdRegex.group(1)
                cmdArgs = cmdRegex.group(2).strip().split()
                if action == "set":
                    if len(cmdArgs) < 2:
                        output("Invalid number of arguments for set command. ", colors["error"])
                        instructions()
                        continue
                    if cmdArgs[0] in settings:
                        currentSettingValue = settings[cmdArgs[0]]
                        output(
                            "Current Value of {}: {}     Changing to: {}".format(
                                cmdArgs[0], currentSettingValue, cmdArgs[1]
                            )
                        )
                        settings[cmdArgs[0]] = cmdArgs[1]
                        output("Save config file?", colors["query"])
                        output(
                            "Saving an invalid option will corrupt file! ", colors["error"]
                        )
                        if (
                                input_line(
                                    "y/n? >",
                                    colors["selection-prompt"],
                                    colors["selection-value"],
                                )
                                == "y"
                        ):
                          try:
                            with open("config.ini", "w", encoding="utf-8") as file:
                                config.write(file)
                          except IOError:
                            output("Permission error! Changes will not be saved for next session.", colors["error"])                      
                    else:
                        output("Invalid setting", colors["error"])
                        instructions()

                elif action == "menu":
                    if input_bool("Do you want to save? (y/N): ", colors["query"], colors["user-text"]):
                        save_story(story)
                    story = None
                    context = None
                    prompt = None
                    break

                elif action == "restart":
                    output("Restarting story...", colors["loading-message"])
                    if len((context + prompt).strip()) == 0:
                        output("Story has no prompt or context. Please enter a valid prompt. ", colors["error"])
                        continue
                    story = new_story(generator, story.context, prompt)

                elif action == "quit":
                    if input_bool("Do you want to save? (y/N): ", colors["query"], colors["user-text"]):
                        save_story(story)
                    exit()

                elif action == "help":
                    instructions()

                elif action == "print":
                    use_wrap = input_bool("Print with wrapping? (y/N): ", colors["query"], colors["user-text"])
                    use_color = input_bool("Print with colors? (y/N): ", colors["query"], colors["user-text"])
                    output("Printing story...", colors["message"])
                    story.print_story(wrap=use_wrap, color=use_color)

                elif action == "retry":
                    if len(story.actions) < 2:
                        output("Restarting story...", colors["loading-message"])
                        if len((context + prompt).strip()) == 0:
                            output("Story has no prompt or context. Please enter a valid prompt. ", colors["error"])
                            continue
                        story = new_story(generator, story.context, prompt)
                        continue
                    else:
                        newaction = story.actions[-1]
                        story.revert()
                        result = story.act(newaction)
                        if story.is_looping():
                            story.revert()
                            output("That action caused the model to start looping. Try something else instead. ",
                                   colors["error"])
                            continue
                        story.print_last()

                elif action == "revert":
                    if len(story.actions) < 2:
                        output("You can't go back any farther. ", colors["error"])
                        continue
                    story.revert()
                    output("Last action reverted. ", colors["message"])
                    story.print_last()

                elif action == "alter":
                    story.results[-1] = alter_text(story.results[-1])
                    story.print_last()

                elif action == "context":
                    story.context = alter_text(story.context)
                    story.print_last()

                elif action == "remember":
                    memory = cmdRegex.group(2).strip()
                    if len(memory) > 0:
                        memory = re.sub("^[Tt]hat +(.*)", "\\1", memory)
                        memory = memory.strip('.')
                        memory = memory.strip('!')
                        memory = memory.strip('?')
                        story.memory.append(memory.capitalize() + ".")
                        output("You remember " + memory + ". ", colors["message"])
                    else:
                        output("Please enter something valid to remember. ", colors["error"])

                elif action == "forget":
                    while True:
                        i = 0
                        output("Select a memory to forget: ", colors["menu"])
                        list_items(story.memory + ["(Finish)"], colors["menu"])
                        i = input_number(len(story.memory))
                        if i == len(story.memory):
                            break
                        else:
                            del story.memory[i]

                elif action == "save":
                    save_story(story)

                elif action == "load":
                    story, context, prompt = load_story()
                    if story:
                        output("Loading story...", colors["message"])
                        story.print_story()

                else:
                    output("Invalid command: " + action, colors["error"])

            # Otherwise this is just a normal action.
            else:
                if act_alts > 0:
                    # Options to select a suggestion action
                    if action in [str(i) for i in range(len(suggested_actions))]:
                        action = suggested_actions[int(action)]

                original_action = action
                action = action.strip()
                # TODO debug stuff to delete
                if action != original_action:
                    logger.debug("STRIPPED WHITE SPACE OFF ACTION %r vs %r", original_action, action)

                # Crop actions to a max length
                # action = action[:4096]

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
                        logger.info("%r. %r, %r", action, any(action.lstrip().startswith(t) for t in ['"', "'"]),
                                    settings.getboolean("action-d20"))
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

                if use_ptoolkit():
                    action = ptprompt("For REAL: ", default="%s" % action)

                output("> " + format_result(action), colors["transformed-user-text"])

                result = story.act(action)

                if story.is_looping():
                    story.revert()
                    output("That action caused the model to start looping. Try something else instead. ",
                           colors["error"])

                if player_won(result):
                    output(result, colors["ai-text"])
                    output("YOU WON. CONGRATULATIONS", colors["error"])
                    list_items(["Start a New Game", "\"I'm not done yet!\" (If you still want to play)"])
                    choice = input_number(1)
                    if choice == 0:
                        break
                    else:
                        output("Sorry about that...where were we?", colors["query"])

                elif player_died(result):
                    output(result, colors["ai-text"])
                    output("YOU DIED. GAME OVER", colors["error"])
                    list_items(["Start a New Game", "\"I'm not dead yet!\" (If you didn't actually die)"])
                    choice = input_number(1)
                    if choice == 0:
                        break
                    else:
                        output("Sorry about that...where were we?", colors["query"])
                        
                output(result, colors["ai-text"])


# This is here for rapid development, without reloading the model. You import play into a jupyternotebook with autoreload
if __name__ == "__main__":
    with open(Path("interface", "clover"), "r", encoding="utf-8") as file:
        print(file.read())
    generator = get_generator()
    play(generator)
