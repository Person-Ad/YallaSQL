# TODOs
- [ ] make expression take & return batches
- [ ] streams
- [ ] ```SQL GROUP BY```
- [ ] constant memory in `BoundValue`


- [ ] add CPU_PAGEABLE in `DeviceType` and `CacheManager` -> low priority

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

# After Streaming & Optimization
 02:38   yallaSQL λ  ```SELECT I, v, (I/2) as half, (I%2) as rem, 2.5 + v from (SELECT id as I, views as v from table_1);```
 COARSENING_FACTOR 5 | BLOCK_DIM 256
small table
Mean: 476ms, Median: 480 ms, Min: 455 ms, Max: 491 ms, StdDev: 11.2428 ms
large table
Mean: 9110.2ms, Median: 9144 ms, Min: 8979 ms, Max: 9213 ms, StdDev: 72.7733 ms
---
 COARSENING_FACTOR 2 | BLOCK_DIM 256
small table
Mean: 462.1ms, Median: 461 ms, Min: 450 ms, Max: 477 ms, StdDev: 8.89326 ms
large table
Mean: 8701.1ms, Median: 8710 ms, Min: 8543 ms, Max: 8863 ms, StdDev: 104.172 ms
---
 COARSENING_FACTOR 2 | BLOCK_DIM 512
small table
Mean: 471.4ms, Median: 472 ms, Min: 456 ms, Max: 484 ms, StdDev: 8.87919 ms
large table
Mean: 8835.2ms, Median: 8830 ms, Min: 8769 ms, Max: 8931 ms, StdDev: 48.2634 ms
---
 COARSENING_FACTOR 5 | BLOCK_DIM 256
small table
Mean: 490.4ms, Median: 490 ms, Min: 479 ms, Max: 508 ms, StdDev: 7.91454 ms
---
 COARSENING_FACTOR 1 | BLOCK_DIM 256
small table
Mean: 459.7ms, Median: 462 ms, Min: 442 ms, Max: 473 ms, StdDev: 9.089 ms
---
 COARSENING_FACTOR 1 | BLOCK_DIM 128
small table
Mean: 479.8ms, Median: 482 ms, Min: 458 ms, Max: 490 ms, StdDev: 8.29216 ms