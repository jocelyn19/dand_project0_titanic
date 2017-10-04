#audit.py
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint
OSM_FILE = "france-finistere-sud_export.osm" 
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
expected = ["Concarneau", "Rue", "Impasse", "Avenue", "Pont", "Boulevard", "Porte", "Route", "Chemin"] #expected names in the dataset
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

#Array to update with the errors encountered in the file
mapping = {"bretagne": "Bretagne",
           "Av.": "Avenue",
           "pont": "Pont",
           "rue": "Rue",
           "Bd" : "Boulevard",
           "bd" : "Boulevard",
           "Pt" : "Pont",
           "Rte": "Route",
           "place": "Place",
           "quai": "Quai",
           "chemin": "Chemin"
           }

# Search string for the special characters and compare the result to the expected list. If not inside, then add it to the street_type array.
def audit_street(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

# Check streetname
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

# Audit function
def audit(osmfile):
    osm_file = open(osmfile, encoding="utf8")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street(street_types, tag.attrib['v'])

    return street_types

 # reformat string to first letter capital, except for abreviation all in capital
def string_case(s):
    if s.isupper():
        return s
    else:
        return s.title()

# update name function
def update_name(name, mapping):
    name = name.split(' ')
    for i in range(len(name)):
        if name[i] in mapping:
            name[i] = mapping[name[i]]
            #reformat
            name[i] = string_case(name[i])
        else:
            name[i] = string_case(name[i])

    name = ' '.join(name)


    return name
print("Get the list of existing name")
print("##############################")
pprint.pprint(dict(audit(OSM_FILE))) # print the existing names

print()
print("List of items with proposed correction:")

st_types = audit(OSM_FILE)

# print the updated names
for street_type, ways in st_types.items():
    for name in ways:
        better_name = update_name(name, mapping)
        print (name, "=>", better_name)
