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
        nodes.add(start[0])
        if (dict.get(start[0], -1) is -1):
            dict[start[0]] = numnode
            numnode += 1
        nodes.add(start[1])
        if (dict.get(start[1], -1) is -1):
            dict[start[1]] = numnode
            numnode += 1
linenum2 = 1

for line in fileinput.input(sys.argv[1], inplace = 1):
    start = line.split()
    if linenum2 > 1:
        line = ""
        line += str(dict[start[0]]) #charity
        line += ' '
        line += str(dict[start[1]]) #donor
        line += ' '
        line += '1'
        line += '\n'
        print line,
    else:
        linenum2 += 1
        print line,
