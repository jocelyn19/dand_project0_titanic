#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code stored in sample_file.py

import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow

"""

    This script will extract a sample of the OSM_FILE and save it under SAMPLE_FILE

"""
OSM_FILE = "france-finistere-sud_export.osm"
SAMPLE_FILE = "france-finistere-sud_export_sample.osm"

k = 10 # Parameter: take every k-th top level element

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """
        Yield element if it is the right type of tag
        Reference:
        http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
        
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


with open(SAMPLE_FILE, 'w') as output:
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write('<osm>\n  ')
    # Write every kth top level element
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % k == 0:
            output.write(str(ET.tostring(element, encoding='utf-8')))

    output.write('</osm>')
    print("sample done")
