import os
import random
from tqdm.auto import tqdm
from faker import Faker
from datetime import datetime, timedelta
import numpy as np
import pandas as pd 

# Initialize Faker instance
fake = Faker()

# Column types and their possible names
column_types = ['int', 'float', 'text', 'datetime']
column_name_dict = {
    'int': [
        'age', 'quantity', 'count', 'score', 'rating', 'id', 'height', 'weight', 
        'population', 'level', 'position', 'rank', 'attendance', 'total_items',
    ],
    'float': [
        'price', 'amount', 'balance', 'total', 'cost', 'discount', 'tax', 'interest', 
        'savings', 'inflation', 'revenue', 'investment', 'currency'
    ],
    'text': [
        'name', 'description', 'address', 'city', 'email', 'comment', 'product_name', 
        'feedback', 'message', 'notes',   'status', 'details', 'subject', 'title'
    ],
    'datetime': [
        'created_at', 'updated_at', 'birth_date', 'order_date', 'delivery_date', 
        'transaction_date', 'due_date', 'last_modified', 'submitted_at', 'completion_date'
    ]
}

flattened = [(column_type, column_name) for column_type in column_types for column_name in column_name_dict[column_type]]


def generate_random_values(column_type):
    """Generate random data based on the column schema and any foreign key relationships."""
    if column_type == 'int':
        return random.randint(1, 1000)
    elif column_type == 'float':
        return round(random.uniform(1.0, 1000.0), 2)
    elif column_type == 'text':
        return fake.text(max_nb_chars=50)
    elif column_type == 'datetime':
        return fake.date_time_this_century()


def generate_unique_values(datatype, n):
    """Efficiently generate n unique values based on the specified datatype."""
    if datatype == 'int':
        return random.sample(range(1, 10 * n), n)  # sample guarantees uniqueness

    elif datatype == 'float':
        values = set()
        while len(values) < n:
            samples = np.round(np.random.uniform(1.0, 1000.0 * n, n * 10), 2)
            values.update(samples.tolist())
        return list(values)[:n]

    elif datatype == 'text':
        values = set()
        while len(values) < n:
            values.update(fake.text(max_nb_chars=200) for _ in range(n))
        return list(values)[:n]

    elif datatype == 'datetime':
        start_date = datetime(1000, 1, 1)
        end_date = datetime(2025, 1, 1)
        delta = (end_date - start_date).days

        unique_days = random.sample(range(delta), n)
        return [start_date + timedelta(days=day) for day in unique_days]

def generate_random_schema(prev_primary_keys: dict):
    columns = [("int", "id (P)")]
    # choose columns randomly
    columns += random.sample(flattened, k = np.random.randint(3, 15))
    
    # choose forigen keys randomly
    if len(prev_primary_keys) != 0:
        num_fk      = np.random.randint(0, len(prev_primary_keys)) # num of forigen keys
        cols_fk     = np.random.choice(list(prev_primary_keys.keys()), size=(num_fk))
    else:
        cols_fk = []
    # shuffle all columns to not have specific order
    return columns, list(cols_fk)
    

def create_csv_file(file_path, num_records, schema, table_name, foreign_keys, prev_primary_keys):
    """Create a CSV file for a table with random data based on the schema and foreign key relationships."""
    data = {}
    for column_type, column_name in tqdm(schema):
        if column_name.endswith("(P)"):
            rows = generate_unique_values(column_type, num_records)
            prev_primary_keys[table_name + "_" + column_name[:-4]] = rows
        else:
            rows = [generate_random_values(column_type) for _ in range(num_records)]
        data[column_name] = rows
    
    for column_name in foreign_keys:
        data[column_name] = random.choices(prev_primary_keys[column_name], k=num_records)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print("finished generating ", table_name)


def create_random_tables(seed, folder_path, num_tables=5):
    """Create multiple random CSV tables with random schemas and relationships."""
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    prev_primary_keys = {} # colname: values # so we can choose from them 
    
    for i in range(num_tables):
        table_name = f"table_{i + 1}"
        schema, foreign_keys = generate_random_schema(prev_primary_keys)
        
        file_path = os.path.join(folder_path, f"{table_name}.csv")
        num_records = np.random.randint(100_000)
        create_csv_file(file_path, num_records, schema, table_name, foreign_keys, prev_primary_keys)

        # prev_primary_keys.extend(primary_keys)

# Example usage
seed = 42  # You can change the seed value
folder_path = 'dataset'
create_random_tables(seed, folder_path)
