import csv
import os

# Define the SSB schema with your naming convention
ssb_schema = {
    'lineorder': [
        'orderkey (N) (P)', 'linenumber (N) (P)', 'customer_custkey (N)', 'part_partkey (N)',
        'supplier_suppkey (N)', 'date_datekey_orderdate (D)', 'orderpriority (T)', 'shippriority (T)',
        'quantity (N)', 'extendedprice (N)', 'ordtotalprice (N)', 'discount (N)',
        'revenue (N)', 'supplycost (N)', 'tax (N)', 'date_datekey_commitdate (D)', 'shipmode (T)'
    ],
    'customer': [
        'custkey (N) (P)', 'name (T)', 'address (T)', 'city (T)', 'nation (T)',
        'region (T)', 'phone (T)', 'mktsegment (T)'
    ],
    'supplier': [
        'suppkey (N) (P)', 'name (T)', 'address (T)', 'city (T)', 'nation (T)',
        'region (T)', 'phone (T)'
    ],
    'part': [
        'partkey (N) (P)', 'name (T)', 'mfgr (T)', 'category (T)', 'brand1 (T)',
        'color (T)', 'type (T)', 'size (N)', 'container (T)'
    ],
    'date': [
        'datekey (N) (P)', 'date (D)', 'dayofweek (T)', 'month (T)', 'year (N)',
        'yearmonthnum (N)', 'yearmonth (T)', 'daynuminweek (N)', 'daynuminmonth (N)',
        'daynuminyear (N)', 'monthnuminyear (N)', 'weeknuminyear (N)', 'sellingseason (T)',
        'lastdayinweekfl (N)', 'lastdayinmonthfl (N)', 'holidayfl (N)', 'weekdayfl (N)'
    ]
}

# Directory containing .tbl files
tbl_dir = './benchmark/ssb-dbgen'  # Adjust to your directory
output_dir = './ssb-benchmark'  # Directory for CSV output

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each table
for table_name, headers in ssb_schema.items():
    tbl_file = os.path.join(tbl_dir, f'{table_name}.tbl')
    csv_file = os.path.join(output_dir, f'{table_name}.csv')

    if not os.path.exists(tbl_file):
        print(f"Table file {tbl_file} not found, skipping.")
        continue

    print(f"Converting {tbl_file} to {csv_file}")

    with open(tbl_file, 'r', encoding='utf-8') as infile, \
         open(csv_file, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write(",".join([f'"{h}"' for h in headers]) + "\n")
        # fix remove last "," in row
        outfile.write("\n".join([l[:-2] for l in infile.readlines()]))

print("Conversion complete.")