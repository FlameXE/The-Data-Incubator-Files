__author__ = 'Havoc'

import re
import networkx as nx
import itertools
from operator import itemgetter

line_replace = ['Chairs', 'Tommy Hilfiger', 'Palm Beach', 'Senior Vice', 'Executive Vice', 'OSL Board Member', 'Special Surgery', 'Gala Co', 'Event Co', 'The', 'Dr.', 'President', 'Major', 'Director', 'His', 'Her', 'Their', 'Imperial', 'Highness', 'Prince', 'Princess', 'Lady', 'CEO', 'Mayor', 'New York', 'New York City']
namelist = []
namelist_seperated = []
namelist_combinations= []

f = open('captions.txt', 'r')

lines = f.readlines()
g = nx.Graph()

x = 0
y = 0
z = 0
a = 0

for line in lines:
    for line_replaces in line_replace:
        line = re.sub(line_replaces,'',line)
    namelist.append(line)

while x < 15:
    namelist_seperated.append(re.findall("[A-Z]['?-?A-Za-z]+ (?:de)? ?[A-Z]['?-?A-Za-z]+(?: (?:de)? ?[A-Z]['?-?A-Za-z]+)?", namelist[x]))
    x += 1

while y < len(namelist_seperated):
    namelist_combinations.append(list(itertools.combinations(namelist_seperated[y], 2)))
    y += 1

while z < len(namelist_combinations):
    while a < len(namelist_combinations[z]):
        g.add_edge(namelist_combinations[z][a][0], namelist_combinations[z][a][1])
        try:
            g[namelist_combinations[z][a][0]][namelist_combinations[z][a][1]]['weight'] += 1
        except:
            g[namelist_combinations[z][a][0]][namelist_combinations[z][a][1]]['weight'] = 1
        a += 1
    a = 0
    z += 1

#print sorted(g.degree_iter(weight = 'weight'),key=itemgetter(1),reverse=True)
#print sorted(nx.pagerank(g, alpha=0.85).items(),key=itemgetter(1),reverse=True)
#print sorted(g.edges(data=True), key=lambda (source,target,data): data['weight'], reverse = True)