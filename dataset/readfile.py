#python script to format the charity_net_graph data

import fileinput
import sys
from sets import Set

if (len(sys.argv)) != 2:
    print 'Usage: python readfile.py filename'
    sys.exit()

linenum = -1 # ignore the table header
nodes = Set()
dict = {}
numnode = 0

for line in fileinput.input(sys.argv[1]):
    linenum += 1
    if linenum > 0:
        start = line.split()
        nodes.add(start[1])
        if (dict.get(start[1], -1) is -1):
            dict[start[1]] = numnode
            numnode += 1
        nodes.add(start[2])
        if (dict.get(start[2], -1) is -1):
            dict[start[2]] = numnode
            numnode += 1
linenum2 = 1

for line in fileinput.input(sys.argv[1], inplace = 1):
    start = line.split()
    if linenum2 > 1:
        line = ""
        line += str(dict[start[1]]) #charity
        line += ' '
        line += str(dict[start[2]]) #donor
        line += ' '
        line += start[3] #date
        line += ' '
        line += start[4] #time of day
        line += ' '
        line += start[5] #amount of money
        line += ' '
        line += start[1]
        line += '-'
        line += start[2]
        line += ':'
        for i in range(6, len(start)):
            line += start[i]
        line += '\n'
        print line,
    else:
        linenum2 += 1
        line = ""
        line += str(linenum)
        line += '\n'
        line += str(len(nodes))
        line += '\n'
        print line,
