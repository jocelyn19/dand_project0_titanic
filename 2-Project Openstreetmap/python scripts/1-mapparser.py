#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Map parser to count the number of tags of the OSMFILE
mapparser.py
"""

import xml.etree.cElementTree as ET
import pprint
OSM_FILE = "france-finistere-sud_export.osm" 
def count_tags(filename):
    tags = {}
    for event, elem in ET.iterparse(filename):
        if elem.tag in tags:
            tags[elem.tag] += 1
        else:
            tags[elem.tag] = 1
    return tags

pprint.pprint(count_tags(OSM_FILE))
