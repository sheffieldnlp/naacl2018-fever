import curses
import json
import sys
from curses.textpad import rectangle

#source activate pytorch; cd Documents/PhD/projects/10\ FEVER/fever-baselines; python src/scripts/review_screen.py data/dump1.json

def render(item):
    stdscr.addstr(2,3,"Claim:")
    stdscr.addstr(2,10, item["text"])
    rectangle(stdscr,1,1,3,100)


    stdscr.addstr(4,3,"Entity:")
    stdscr.addstr(4,10,item["page"])

    stdscr.addstr(4,51,"Oracle:")
    stdscr.addstr(4,58,str(item["isOracle"]))

    stdscr.addstr(4, 76, "Reval:")
    stdscr.addstr(4, 82, str(item["isReval"]))

    rectangle(stdscr,3,1,5,48)
    rectangle(stdscr, 3, 49, 5, 73)
    rectangle(stdscr, 3, 74, 5, 100)

    stdscr.addstr(6,3,"Annotation:")
    stdscr.addstr(6,13,item["verifiable"])

    stdscr.addstr(6,43,"Label:")
    stdscr.addstr(6,53,item["label"] if "label" in item else "")

    rectangle(stdscr, 5, 1, 7, 41)
    rectangle(stdscr, 5, 42, 7, 100)



    stdscr.refresh()

# Read dump
with open(sys.argv[1],"r") as f:
    data = json.load(f)

# Make curses
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()



for item in data:
    render(item)
    break

