import re
from getconfig import settings
from gpt2generator import GPT2Generator

#The game interface code should be kept seperate from the story class
class Story:
    #the initial prompt is very special.
    #We want it to be permanently in the AI's limited memory (as well as possibly other strings of text.)
    def __init__(self, MC, prompt='', numResults=1):
        self.numResults=numResults
        self.story = []
        self.longTermMemory = []
        self.prompt = prompt
        self.storyGen = GPT2Generator(MC)
        self.sugGen = GPT2Generator(MC)
        self.settings()

    def settings(self):
        self.storyGen.temperature = settings.getfloat('temp')
        self.storyGen.top_p = settings.getfloat('top-p')
        self.storyGen.top_k = settings.getint('top-keks')
        self.storyGen.repetition_penalty = settings.getfloat('rep-pen')
        self.storyGen.generate_num = settings.getint('generate-num')

        self.sugGen.temperature = settings.getfloat('action-temp')
        self.sugGen.top_p = settings.getfloat('top-p')
        self.sugGen.top_k = settings.getint('top-keks')
        self.sugGen.repetition_penalty = settings.getfloat('rep-pen')
        self.sugGen.generate_num = settings.getint('action-generate-num')

    def act(self, action):
        assert(self.prompt+action)
        results=[]
        for i in range(self.numResults):
            assert(settings.getint('top-keks') is not None)
            results.append(self.storyGen.generate(self.getStory()+action, self.prompt))
            
        self.story.append([action, results])
        return results

    #only relevant when multiple results are supported
    #this will be added in the future, don't remove it
    #chosen result is always placed first, simplifies many things
    def chooseResult(self, num):
       results=self.story[1] 
       best=results.pop(num)
       results.insert(0,best)

    #Results 
    def getStory(self):
        lines = []
        for line in self.story:
            lines.append(line[0])
            lines.append(line[1][0])
        return '\n\n'.join(lines)

    def getSuggestion(self):
        #temporary fix (TODO)
        return re.sub('\n.*', '', self.sugGen.generate(self.getStory()+"\n\n> You", self.prompt))

    def __str__(self):
        return self.prompt+self.getStory()

#    def save()
#        file=Path('saves', self.filename)
