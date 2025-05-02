# TODOs
- [x] streams
- [ ] add strings
- [ ] ```SQL WHERE```
    - [ ] compare for strings -> I think it can be done by filling zeros
- [ ] ```SQL SORT BY```
- [ ] ```SQL Inner join```

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

# My Logs
## Binary Operators
after streaming & optimizations
```
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

> after fixing/changining aligment from 32 to BLOCK_DIM so I load batches within specific size "16mb" & aligned to 

---
COARSENING_FACTOR 2 | BLOCK_DIM 256 | Max Bytes ber batch "32mb"
Mean: 441.5ms, Median: 441 ms, Min: 429 ms, Max: 453 ms, StdDev: 7.00357 ms
---
COARSENING_FACTOR 2 | BLOCK_DIM 256 | Max Bytes ber batch "16mb"

for small table:
1. Mean: 447.7ms, Median: 450 ms, Min: 440 ms, Max: 460 ms, StdDev: 7.12811 ms
2. Mean: 433.8ms, Median: 435 ms, Min: 424 ms, Max: 443 ms, StdDev: 6.43117 ms
3. Mean: 437ms, Median: 435 ms, Min: 426 ms, Max: 456 ms, StdDev: 8.48528 ms

for large table
1. Mean: 8705ms, Median: 8667 ms, Min: 8578 ms, Max: 8946 ms, StdDev: 119.172 ms
---
COARSENING_FACTOR 2 | BLOCK_DIM 512 | Max Bytes ber batch "16mb"
Mean: 678.5ms, Median: 678 ms, Min: 663 ms, Max: 704 ms, StdDev: 11.0023 ms

> after fixing calculateOptimalBatchSize: send real selected dtypes of columns + correct returning

COARSENING_FACTOR 5 | BLOCK_DIM 256 | Max Bytes ber batch "16mb"
1. Mean: 448.5ms, Median: 443 ms, Min: 433 ms, Max: 486 ms, StdDev: 14.5413 ms
2. Mean: 433.2ms, Median: 430 ms, Min: 421 ms, Max: 468 ms, StdDev: 12.867 ms
3. Mean: 436.2ms, Median: 440 ms, Min: 420 ms, Max: 461 ms, StdDev: 13.1666 ms
 
for large table
1. Mean: 8194.4ms, Median: 8211 ms, Min: 8051 ms, Max: 8413 ms, StdDev: 109.692 ms

---

COARSENING_FACTOR 5 | BLOCK_DIM 512 | Max Bytes ber batch "16mb"
1. Mean: 434.3ms, Median: 435 ms, Min: 422 ms, Max: 466 ms, StdDev: 12.042 ms
2. Mean: 423.3ms, Median: 425 ms, Min: 410 ms, Max: 437 ms, StdDev: 8.18596 ms
3. Mean: 424.4ms, Median: 425 ms, Min: 416 ms, Max: 430 ms, StdDev: 3.747 ms
4. Mean: 414.5ms, Median: 414 ms, Min: 407 ms, Max: 423 ms, StdDev: 4.58803 ms

for large table
1. Mean: 8553.1ms, Median: 8526 ms, Min: 8210 ms, Max: 8961 ms, StdDev: 265.411 ms
2. Mean: 8183.5ms, Median: 8150 ms, Min: 8053 ms, Max: 8489 ms, StdDev: 137.61 ms

--- 
> after adding `cudaFree(0)` when the program start
1. Mean: 434.7ms, Median: 426 ms, Min: 414 ms, Max: 474 ms, StdDev: 19.4373 ms
2. Mean: 418.9ms, Median: 419 ms, Min: 406 ms, Max: 430 ms, StdDev: 7.06329 ms
3. Mean: 412.7ms, Median: 414 ms, Min: 404 ms, Max: 421 ms, StdDev: 5.55068 ms

---
> after pin buffer or reading & -O3 in building
small table
1. Mean: 87.9ms, Median: 85 ms, Min: 74 ms, Max: 109 ms, StdDev: 8.9493 ms
2. Mean: 76.9ms, Median: 80 ms, Min: 71 ms, Max: 83 ms, StdDev: 4.50444 ms
large table
1. Mean: 702.8ms, Median: 699 ms, Min: 689 ms, Max: 732 ms, StdDev: 13.0445 ms
2. Mean: 723.3ms, Median: 701 ms, Min: 691 ms, Max: 836 ms, StdDev: 44.8733 ms
3. Mean: 719.2ms, Median: 723 ms, Min: 701 ms, Max: 751 ms, StdDev: 16.6721 ms
```

> after free intermidate results :xd

```
small table
1.  Mean: 112.9ms, Median: 110 ms, Min: 93 ms, Max: 132 ms, StdDev: 11.4756 ms
2.  Mean: 124.7ms, Median: 124 ms, Min: 111 ms, Max: 149 ms, StdDev: 13.2216 ms
3.  Mean: 111.7ms, Median: 113 ms, Min: 101 ms, Max: 125 ms, StdDev: 7.65572 ms
4.  Mean: 105.8ms, Median: 107 ms, Min: 96 ms, Max: 111 ms, StdDev: 4.70744 ms

large table
1.  Mean: 870ms, Median: 867 ms, Min: 834 ms, Max: 917 ms, StdDev: 27.0629 ms
2.  Mean: 857.6ms, Median: 848 ms, Min: 818 ms, Max: 954 ms, StdDev: 38.103 ms
```

> may be add Garbage Collector to free without making writing wait


## Filters
```SQL
SELECT weight from table_1 WHERE views<id;
```
Mean: 88.4ms, Median: 85 ms, Min: 83 ms, Max: 97 ms, StdDev: 5.27636 ms

```SQL
SELECT views, 2*views as double, id from table_1 WHERE 2*views<id;
```
Mean: 84.3ms, Median: 83 ms, Min: 73 ms, Max: 103 ms, StdDev: 7.00071 ms

ssb-benchmark
SELECT suppkey, name, region FROM supplier WHERE region='AMERICA';
Mean: 43.4ms, Median: 43 ms, Min: 42 ms, Max: 45 ms, StdDev: 0.916515 ms
Mean: 42.4ms, Median: 44 ms, Min: 33 ms, Max: 44 ms, StdDev: 3.2 ms

```SQL 
SELECT views, 2*views as double_views, id, weight from table_1 WHERE address='Long career air now success.\\nGas author any claim on even buy of. Particularly move nothing movement build. Drop oil lose let sit be.\\nRepublican trouble firm adult black pressure music. Project talk official difference old answer.' and id<3000
```

small table
1. Mean: 100.1ms, Median: 104 ms, Min: 93 ms, Max: 107 ms, StdDev: 5.68243 ms
2. Mean: 99ms, Median: 98 ms, Min: 93 ms, Max: 107 ms, StdDev: 5.56776 ms
3. Mean: 98.9ms, Median: 96 ms, Min: 93 ms, Max: 107 ms, StdDev: 5.14684 ms

large_table (since no writing or anything except checking)
Mean: 795.9ms, Median: 799 ms, Min: 752 ms, Max: 832 ms, StdDev: 27.6892 ms

```SQL 
SELECT id, updated_at FROM table_1 WHERE updated_at<'2009-01-02 12:01:17';
```
small table
1. Mean: 86.9ms, Median: 84 ms, Min: 81 ms, Max: 105 ms, StdDev: 7.07743 ms

large table
1.  Mean: 753.5ms, Median: 740 ms, Min: 720 ms, Max: 817 ms, StdDev: 33.3504 ms


```SQL
SELECT id, updated_at FROM table_1 WHERE updated_at<'2009-01-02 12:01:17' and id < 3000 or profit_margin*views < 500.23;
```
small table
1. Mean: 98.7ms, Median: 98 ms, Min: 82 ms, Max: 110 ms, StdDev: 7.96304 ms
large table
1. Mean: 759.7ms, Median: 748 ms, Min: 722 ms, Max: 856 ms, StdDev: 40.7874 ms