The various trials and tribulations I have encountered trying to move my lovely bug-free python script to a vanilla windows machine:

- There is NOTHING remotely friendly or recognizable about the windows shell. It's not unix or linux out of the box, apparently. I spent a good two hours trying to install a bash shell before realizing that it was probably a waste of my time seeing that most of the standard command line prompts (or at least the one I needed - cd) would still work.

It turns out that my dad's empty husk of a computer actually has python... which I didn't realize until I naively tried to run my script: "python similarity_analysis.py". Oh joy. But then I discovered...

- No python packages. My smart self thought that meant I needed to create a windows executable file, so I downloaded py2exe and pyinstaller on both my machine and my dad's before realizing-- oops-- you can't make an executable unless it's for the operating machine you're currently using. Which seems ridiculous but hey I don't make the rules. Good God. I looked at Wine briefly before suddenly realizing that I can literally just pip install exactly the packages that were giving me an error. So I pip installed and ran my script ten billion times until all the packages were installed.

- Oh my God, the script was running. Everything was looking good. My heart started to soar and I actually laughed out loud before God decided to punish me for my arrogance: 
```UnicodeEncodeError: 'charmap' codec can't encode characters in position xx: character maps to undefined```
WHY GOD, WHY?? So after meticulously scraping through StackOverflow I've discovered that while the default encoder for linux/unix is utf-8, windows uses this 'charmap' thing which doesn't do as good of a job. So either I need to change the way my script reads in the files or I need to change something on that godforsaken PC to make it use utf-8 instead of that-encoding-which-shall-not-be-named. Which leads me to the next issue:

- Do I REALLY have to download a text editor to use for the PC's command line? Why why WHY on Earth wouldn't there already be something for me to use? It's bad enough that I've been hopping back and forth between a PC and a Mac. Maybe the investment of downloading a text editor for the PC will be a good one... because going back and forth right now is giving me some kind of a terminal disease.  

- Going back to the Unicode thing, it turns out this is an actual issue with lots of documentation [here](https://stackoverflow.com/questions/5419/python-unicode-and-the-windows-console/32176732#32176732), [here](https://stackoverflow.com/questions/878972/windows-cmd-encoding-change-causes-python-crash/3259271), and [here](https://bugs.python.org/issue1602) among other places. Upon taking a break and eating a tangerine... I have discovered the error amounted to no more than specifying ```write(newfile, encode = 'utf-8')``` so that the file it writes out has the correct encoding.

Also update: we have installed vim on the dinosaur PC. It has made things better. 

Progress has been made but the next round of bugs has considerably weakened me. Time to take a break and come back to this later.   
