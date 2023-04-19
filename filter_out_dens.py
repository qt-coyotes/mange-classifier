#!/usr/bin/env python3
import os
import re
import json

Tiziana = re.compile('Tiziana')

with open('data/qt-coyotes-merged.json', 'r') as f:
    js = json.load(f)

js2 = {
    'info': js['info'],
    'categories': js['categories'],
    'images': [],
    'annotations': [],
}

print(len(js['images']))
print(len(js['annotations']))

for image in js['images']:
    if Tiziana.match(image['rights_holder']):
        this_id = image['id']

        for annote in js['annotations']:
            if annote['image_id'] == this_id:
                js2['annotations'].append(annote)
        js2['images'].append(image)

print(len(js2['images']))
print(len(js2['annotations']))

with open('qt-coyotes-toronto.json', 'w') as f:
    json.dump(js2, f, indent=4)
