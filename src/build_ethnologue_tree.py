'''
Script to build Ethnologue language family trees
'''

import json

from anytree import Node
from anytree.exporter import DictExporter
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

lang_codes = []
for c in alphabet:
    my_url = "https://www.ethnologue.com/browse/codes/" + c
    req = Request(my_url, headers={'User-Agent': 'Mozilla/5.0'})
    web_byte = urlopen(req).read()
    webpage = web_byte.decode('utf-8')
    page = soup(webpage, "html.parser")
    table = page.find(lambda tag: tag.name=='table')
    rows = table.find_all('tr')
    for row in rows:
        cols = row.findAll('td')
        for col in cols:
            lang_codes.append(col.text.strip())

all_info = {}
for code in lang_codes:
    print(code)
    info = []
    info.append(code)
    url = "https://www.ethnologue.com/language/" + code
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    web_byte = urlopen(req).read()
    webpage = web_byte.decode('utf-8')
    page = soup(webpage, "html.parser")
    lang_name = page.h1.text
    subgroup = page.findAll("div", {"class":"views-field views-field-field-subgroup"})
    if (len(subgroup)):
        lang_families = page.findAll("div", {"class":"views-field views-field-field-subgroup"})[0].div.text.split('\u203a')
    else:
        lang_families = None
    info.append(lang_name)
    info.append(lang_families)
    all_info[code] = info

ethnologue_info_path = '../metadata/ethnologue_info.json'
with open(ethnologue_info_path, 'w') as f:
    json.dump(all_info, f)

roots = []
visited_fams = {}
for code in all_info:
    info = all_info[code]
    name, families = info[1], info[2]
    parent = None
    if families:
        for fam in families:
            if not parent:
                if fam not in visited_fams:
                    root = Node(fam)
                    visited_fams[fam] = root
                    parent = root
                    roots.append(root)
                else:
                    parent = visited_fams[fam]
            else:
                if fam not in visited_fams:
                    new = Node(fam, parent = parent)
                    visited_fams[fam] = new
                    parent = new
                else:
                    parent = visited_fams[fam]
    leaf = Node(name + str(" ("+code+")"), parent)

adict = []
exporter = DictExporter()
for root in roots:
    adict.append(exporter.export(root))
    
with open('../metadata/ethnologue_forest.json', 'w') as f:
    json.dump(adict, f)
