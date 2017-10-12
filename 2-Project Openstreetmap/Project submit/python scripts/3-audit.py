#audit.py
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
city_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
phone_type_re = re.compile("/^(\+33|0)[1-9]( \d\d){4}$/", re.IGNORECASE)
postcode_type_re = re.compile("\d{2}[ ]?\d{3}", re.IGNORECASE)
expected = ["Concarneau", "Rue", "Impasse", "Avenue", "Pont", "Boulevard", "Porte", "Route", "Chemin"] #expected names in the dataset
expected_postcodes = ["29920", "29900"]
expected_cities = ["Concarneau", "Trégunc"]
expected_phones = ["+33 2 98 50 53 50"]
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
OSM_FILE = "france-finistere-sud_export.osm"

#Array to update with the errors encountered in the file for streetnames
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

# look if the postode has the right format : in France 5 digits
def audit_postcode(postcode_types, postcode):
    m = postcode_type_re.search(postcode)
    if m:
        postcode_type = m.group()
        if postcode_type not in expected_postcodes:
            postcode_types[postcode_type].add(postcode)

# Search string for the special characters and compare the result to the expected list. If not inside, then add it to the city_type array.
def audit_city(city_types, city_name):
    m = city_type_re.search(city_name)
    if m:
        city_type = m.group()
        if city_type not in expected_cities:
            city_types[city_type].add(city_name)

# Search string for the special characters and compare the result to the expected list. If not inside, then add it to the city_type array.
def audit_phone(phone_types, phone_number):
    m = phone_type_re.search(phone_number)
    if m:
        phone_type = m.group()
        if phone_type not in expected_phones:
            phone_types.add(phone_number)

# Check streetname
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

# Check postal code
def is_postcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

# Check cities
def is_city(elem):
    return (elem.attrib['k'] == "addr:city")

# Check phones
def is_phone(elem):
    return (elem.attrib['k'] == "phone")

# Audit function
def audit(osmfile):
    osm_file = open(osmfile, encoding="utf8")
    street_types = defaultdict(set)
    postcode_types = defaultdict(set)
    city_types = defaultdict(set)
    phone_types = defaultdict(set)
    all_phone = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street(street_types, tag.attrib['v'])
                elif is_postcode(tag):
                    audit_postcode(postcode_types, tag.attrib['v'])
                elif is_city(tag):
                    audit_city(city_types, tag.attrib['v'])
                elif is_phone(tag):
                    audit_phone(phone_types, tag.attrib['v'])
                    all_phone['tel'].add(tag.attrib['v'])

    return (street_types, postcode_types, city_types, phone_types, all_phone)

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

# update phone function (remove space in phone numbers)
"""
    To ensure consistency of the phone numbers I will remove all the spaces and parenthesis
    to avoid the following phone numbers:
    +33 12 34 56 67 --> +3312345667
    +33 (1) 12234556 --> +33112234556
"""
def update_phone(phone):
    phone = phone.replace(' ', '')
    phone = phone.replace('(', '')
    phone = phone.replace(')', '')

    return phone

keys = audit(OSM_FILE)

print("Get the list of existing street names")
print("##############################")
pprint.pprint(dict(keys[0])) # print the existing names

print("Get the list of existing post codes")
print("##############################")
pprint.pprint(dict(keys[1])) # print the existing post codes

print("Get the list of existing cities")
print("##############################")
pprint.pprint(dict(keys[2])) # print the existing cities

print("Get the list of existing phones")
print("##############################")
pprint.pprint(dict(keys[3])) # print the existing phones
pprint.pprint(dict(keys[4]).get('tel')) # print the existing phones
print()
print("List of items with proposed correction for streetnames:")

# print the updated streetnames
for name in dict(keys[0]):
        better_name = update_name(name, mapping)
        print (name, "=>", better_name)

print("List of items with proposed corrections for phone numbers:")
# print the updated phone number
for phone in dict(keys[4]).get('tel'):
        better_phone = update_phone(phone)
        print (phone, "=>", better_phone)
