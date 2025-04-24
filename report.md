## Projection & Scan Operator
### 📋 Execution Time Profiling Table
#### 📁 **Table Size: 16MB**

| **Query Type** | **Columns Selected** | **yallaSQL Time (ms)** | **DuckDB Time (ms)** | **Notes** |
|----------------|----------------------|-------------------------|-----------------------|-----------|
| First Execution | `D, C, B, A`        | 1114                    | 4079                  | DuckDB cold start |
| Repeated Exec. | `D, C, B, A`         | 652–797                 | 1539–1560             | yallaSQL consistently faster |
| Single Column   | `A`                 | 535                     | 1403–1559             | yallaSQL outperforms DuckDB |

#### 📁 **Table Size: 750MB**

| **Query Type** | **Columns Selected** | **yallaSQL Time (ms)** | **DuckDB Time (ms)** | **Notes** |
|----------------|----------------------|-------------------------|-----------------------|-----------|
| First Execution | `A`                 | 10496                   | 22799                 | yallaSQL over 2x faster |
| Repeated Exec. | `A`                 | –                       | –                     | No further warm-up runs shown |
| Full Projection | `D, C, B, A`         | 8790                    | 28765                 | yallaSQL ~3x faster |

---
