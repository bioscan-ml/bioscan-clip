#!/usr/bin/env python
#
# Take an BIOSCAN-CLIP results csv and flatten it to one metric per row

import argparse
import csv
import sys

def readCsv(input, delimiter=None):
    if delimiter is None:
        if input.endswith('.tsv'):
            delimiter = '\t'
        else:
            delimiter = ','
    with open(input, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        rows = [r for r in reader]
        return (rows, reader.fieldnames)

def main():
    # Argument processing
    parser = argparse.ArgumentParser(description='Flatten BIOSCAN-CLIP results csv')
    parser.add_argument('-i', '--input',
                        required=True,
                        help='Input file of results')
    parser.add_argument('-o', '--output',
                        required=False,
                        type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='Output file of flattened')
    args = parser.parse_args()
    metric_value_columns = ['Seen_Order','Seen_Family','Seen_Genus','Seen_Species','Unseen_Order','Unseen_Family','Unseen_Genus','Unseen_Species']
    metric_name_column = 'Metric'
    
    rows, fieldnames = readCsv(args.input)
    ignore_fields = [f for f in metric_value_columns]
    ignore_fields.append(metric_name_column)
    keep_fieldnames = [f for f in fieldnames if f not in ignore_fields]
    updated_fieldnames = [f for f in keep_fieldnames]
    updated_fieldnames.extend(['micro_macro','top_k','seen_unseen','taxon','value'])

    writer = csv.DictWriter(args.output, fieldnames=updated_fieldnames)
    writer.writeheader()
    for row in rows:
        new_row = {k:v for k,v in row.items() if k in keep_fieldnames}
        metric_parts = row[metric_name_column].split('_')
        new_row['micro_macro'] = metric_parts[0]
        new_row['top_k'] = metric_parts[1].replace('Top-','')
        for f in metric_value_columns:
            output_row = {k:v for k,v in new_row.items()}
            f_parts = f.split('_')
            output_row['seen_unseen'] = f_parts[0]
            output_row['taxon'] = f_parts[1]
            output_row['value'] = row[f]
            writer.writerow(output_row)
        

if __name__ == "__main__": 
    main()
