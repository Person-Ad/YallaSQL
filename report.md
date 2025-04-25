## Tabkes

#### 📁 **Table Size: 16MB**
- schema:
    ```
    "id (N) (P)","views (N) ","weight (N) ","address (T) ","profit_margin (N) ","altitude (N) ","price (N) ","shares (N) ","category_id (N) ","updated_at (D) "
    ```
#### 📁 **Table Size: 750MB**

- schema:
    ```
    "id (N) (P)","views (N) ","weight (N) ","address (T) ","profit_margin (N) ","altitude (N) ","price (N) ","shares (N) ","category_id (N) ","updated_at (D) ","rating (N) ","score (N) ","attendance (N) ","downloads (N) ","inventory (N) ","memory_usage (N) ","notes (T) ","renewal_date (D) ","city (T) ","retweets (N) ","occupation (T) ","last_name (T) ","bio (T) ","description (T) ","growth_rate (N) ","reviewed_at (D) ","last_seen (D) ","feedback (T) ","timestamp (D) ","university (T) ","age (N) ","domain (T) ","created_at (D) ","order_id (N) ","last_modified (D) ","exchange_rate (N) ","savings (N) "
    ```

## Projection & Scan Operator
```sql
SELECT D, C, B, A FROM (SELECT updated_at as A, price as B, address as C, id as D FROM table_1 );
```

| **Configuration**                              | **Table Size** | **Run 1 (ms)** | **Run 2 (ms)** | **Run 3 (ms)** | **Average (ms)** |
|------------------------------------------------|----------------|----------------|----------------|----------------|------------------|
| **Duckdb**                                     | Small          | 1514           | 1553           | 1362           | 1476.33          |
| **Kernel with BlockDim = 256, Coarsing Factor = 2** | Small          | 559            | 544            | 555            | **552.667**          |
| **Kernel with BlockDim = 256, Coarsing Factor = 5** | Small          | 563            | 565            | 595            | 574.333          |
| **Kernel with BlockDim = 256, Coarsing Factor = 10** | Small          | 657            | 600            | 582            | 613              |
| **Kernel with BlockDim = 256, Coarsing Factor = 2** | Large          | 8240           | 8316           | 8463           | 8339.67          |
| **Kernel with BlockDim = 256, Coarsing Factor = 5** | Large          | 7966           | 7548           | 8042           | **7852**             |
| **Kernel with BlockDim = 256, Coarsing Factor = 10** | Large          | 8425           | 8636           | 8608           | 8556.33          |



---


## Binary Operators
```SQL 
SELECT I, v, (I/2) as half, (I%2) as rem, 2.5*I+v+3 from (SELECT id as I, views as v from table_1);
```

| **Configuration**                              | **Table Size** | **Run 1 (ms)** | **Run 2 (ms)** | **Run 3 (ms)** | **Average (ms)** |
|------------------------------------------------|----------------|----------------|----------------|----------------|------------------|
| **Duckdb**                                     | Small          | 1576           | 1789           | 1690           | 1685             |
| **Without Memory/Streaming Optimization**       |                |                |                |                |                  |
| BLOCK_DIM 256, Coarsing Factor = 2             | Small          | 456            | 491            | 476            | 474.333          |
| BLOCK_DIM 256, Coarsing Factor = 5             | Small          | 423            | 426            | 439            | **429.333**          |
| BLOCK_DIM 256, Coarsing Factor = 10            | Small          | 438            | 442            | 449            | 443              |
| BLOCK_DIM 256, Coarsing Factor = 2             | Large          | 8417           | 8339           | 7675           | 8143.67          |
| BLOCK_DIM 256, Coarsing Factor = 5             | Large          | 7913           | 7233           | 7950           | **7698.67**          |
| BLOCK_DIM 256, Coarsing Factor = 10            | Large          | 7976           | 8003           | 8038           | 8005.67          |


