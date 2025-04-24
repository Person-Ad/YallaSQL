## Projection & Scan Operator
### 📋 Execution Time Profiling Table
- query:
    ```sql
    SELECT D, C, B, A FROM (SELECT updated_at as A, price as B, address as C, id as D FROM table_1 );
    SELECT A FROM (SELECT updated_at as A, price as B, address as C, id as D FROM table_1 );
    ```

#### 📁 **Table Size: 16MB**
- schema:
    ```
    "id (N) (P)","views (N) ","weight (N) ","address (T) ","profit_margin (N) ","altitude (N) ","price (N) ","shares (N) ","category_id (N) ","updated_at (D) "
    ```
- results:
    | **Query Type** | **Columns Selected** | **yallaSQL Time (ms)** | **DuckDB Time (ms)** | **Notes** |
    |----------------|----------------------|-------------------------|-----------------------|-----------|
    | First Execution | `D, C, B, A`        | 1114                    | 4079                  | DuckDB cold start |
    | Repeated Exec. | `D, C, B, A`         | 652–797                 | 1539–1560             | yallaSQL consistently faster |
    | Single Column   | `A`                 | 535                     | 1403–1559             | yallaSQL outperforms DuckDB |

#### 📁 **Table Size: 750MB**

- schema:
    ```
    "id (N) (P)","views (N) ","weight (N) ","address (T) ","profit_margin (N) ","altitude (N) ","price (N) ","shares (N) ","category_id (N) ","updated_at (D) ","rating (N) ","score (N) ","attendance (N) ","downloads (N) ","inventory (N) ","memory_usage (N) ","notes (T) ","renewal_date (D) ","city (T) ","retweets (N) ","occupation (T) ","last_name (T) ","bio (T) ","description (T) ","growth_rate (N) ","reviewed_at (D) ","last_seen (D) ","feedback (T) ","timestamp (D) ","university (T) ","age (N) ","domain (T) ","created_at (D) ","order_id (N) ","last_modified (D) ","exchange_rate (N) ","savings (N) "
    ```

- results:

    | **Query Type** | **Columns Selected** | **yallaSQL Time (ms)** | **DuckDB Time (ms)** | **Notes** |
    |----------------|----------------------|-------------------------|-----------------------|-----------|
    | First Execution | `A`                 | 10496                   | 22799                 | yallaSQL over 2x faster |
    | Repeated Exec. | `A`                 | –                       | –                     | No further warm-up runs shown |
    | Full Projection | `D, C, B, A`         | 8790                    | 28765                 | yallaSQL ~3x faster |

---
