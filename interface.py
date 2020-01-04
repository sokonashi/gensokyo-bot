from getconfig import settings, colors

def boolValue(bool):
	return "on" if bool else "off"

def instructions():
	print('\nAID2: Clover Edition Instructions: \n  Enter actions starting with a verb ex. "go to the tavern" or "attack the orc."\n  To speak enter say "(thing you want to say)" or just "(thing you want to say)"')
	print('The following commands can be entered for any action:')
	print('  "/revert"                Reverts the last action allowing you to pick a different action.')
	print('  "/quit"                  Quits the game and saves')
	print('  "/menu"                  Starts a new game and saves your current one')
	print('  "/retry"                 Retries the last action')
	print('  "/restart"               Restarts the current story')
	print('  "/print"                 Prints a transcript of your adventure (without extra newline formatting)')
	print('  "/alter"                 Edit the last prompt from the AI')
	print('  "/context"               Edit the story\'s permanent context paragraph')
	print('  "/remember [SENTENCE]"   Commits something permanently to the AI\'s memory')
	print('  "/forget"                Opens a menu allowing you to remove permanent memories')
	print('  "/save"                  Saves your game to a file in the game\'s save directory')
	print('  "/load"                  Loads a game from a file in the game\'s save directory')
	print('  "/help"                  Prints these instructions again')
	print('  "/set [SETTING] [VALUE]" Sets the specified setting to the specified value.:')
	print('        temp               Higher values make the AI more random. Default: 0.4 | Current:', settings.getfloat("temp"))
	print('        rep-pen            Controls how repetitive the AI is allowed to be. Default: 1.2 | Current:', settings.getfloat("rep-pen"))
	print('        text-wrap-width    Maximum width of lines printed by computer. Default: 80 | Current:', settings.getint("text-wrap-width"))
	print('        console-bell       Beep after AI generates text? Default: on | Current:', boolValue(settings.getboolean("bell")))
	print('        top-keks           Number of words the AI can randomly choose. Default: 20 | Current:', settings.getint("top-keks"))
	print('        generate-num       Default: 60 | Current:', settings.getint("generate-num"))
	print('        top-p              Default: 0.9 | Current:', settings.getfloat("top-p"))
	print('        log-level          Default: 3 | Current:', settings.getint("log-level"))
	print('        action-sugg        How many actions to generate, 0 is off. Default: 4 | Current:', settings.getint("action-sugg"))
	print('        action-d20         Make actions difficult. Default: on | Current:', boolValue(settings.getboolean("action-d20")))
	print('        action-temp        How random the suggested actions are. Default: 1 | Current:', settings.getfloat("action-temp"), '\033[39m')
