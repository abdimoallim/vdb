### vdb

A lightweight, header-only C library for storing and searching high-dimensional vector embeddings with optional multithreading support.

![](/benchmark.png)

### Features

- Header-only implementation (single file: `vdb.h`)
- Multiple distance metrics (cosine, euclidean, dot product)
- Optional thread-safe operations via `#define VDB_MULTITHREADED`
- Save/load database to/from disk
- Custom memory allocators support
- No dependencies (except `pthreads` for multithreading)
- Python bindings (refer to [`vdb.py`](/vdb.py))

### Usage

```c
/*test.c*/
#include "vdb.h"

int main(void) {
  vdb_database *db = vdb_create(128, VDB_METRIC_COSINE);

  float embedding[128] = { /* ... */ };
  vdb_add_vector(db, embedding, "vec1", NULL);

  float query[128] = { /* ... */ };
  vdb_result_set *results = vdb_search(db, query, 5);

  vdb_free_result_set(results);
  vdb_destroy(db);
  return 0;
}
```

Include [`vdb.h`](/vdb.h) and compile with either approach, `pthreads` is not necessarily available which is why this is behind a flag.

**Single-threaded:**

```bash
gcc -O2 test.c -o test -lm
```

**Multi-threaded:**

```bash
gcc -O2 -DVDB_MULTITHREADED test.c -o test -lpthread -lm
```

### API Reference

#### Database management

| Function | Return Type | Description |
|-|-|-|
| `*vdb_create(size_t dimensions, vdb_metric metric)` | `vdb_database` | Creates a new vector database. |
| `vdb_destroy(vdb_database *db)` | `void` | Frees all resources associated with the database. |
| `vdb_count(const vdb_database *db)` | `size_t` | Returns the number of vectors in the database. |
| `vdb_dimensions(const vdb_database *db)` | `size_t` | Returns the dimensionality of vectors. |

#### Vector operations

| Function | Return Type | Description |
|-|-|-|
| `vdb_add_vector(vdb_database *db, const float *data, const char *id, void *metadata)` | `vdb_error` | Adds a vector to the database with optional ID and metadata. |
| `vdb_remove_vector(vdb_database *db, size_t index)` | `vdb_error` | Removes a vector at the specified index. |
| `vdb_get_vector(const vdb_database \*db, size_t index, float **out_data, char **out_id, void **out_metadata)` | `vdb_error` | Retrieves a vector and its metadata. |

#### Search

| Function | Return Type | Description |
|-|-|-|
| `*vdb_search(const vdb_database *db, const float *query, size_t k)` | `vdb_result_set` | Performs [k-nearest neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) search. Returns NULL on error. |
| `vdb_free_result_set(vdb_result_set *result_set)` | `vdb_result_set` | Frees search results. |

#### Persistence

| Function | Return Type | Description |
|-|-|-|
| `vdb_save(const vdb_database *db, const char *filename)` | `vdb_error` | Saves the database to disk. |
| `*vdb_load(const char *filename)` | `vdb_database` | Loads a database from disk. |

### Distance metrics

| Metric | Description |
|-|-|
| `VDB_METRIC_COSINE` | Cosine distance (1 - cosine similarity) |
| `VDB_METRIC_EUCLIDEAN` | Euclidean (L2) distance |
| `VDB_METRIC_DOT_PRODUCT` | Negative dot product |

### Error codes

| Error code | Label |
|-|-|
| `0` | `VDB_OK` |
| `-1` | `VDB_ERROR_NULL_POINTER` |
| `-2` | `VDB_ERROR_INVALID_DIMENSIONS` |
| `-2` | `VDB_ERROR_OUT_OF_MEMORY` |
| `-2` | `VDB_ERROR_NOT_FOUND` |
| `-2` | `VDB_ERROR_INVALID_INDEX` |
| `-2` | `VDB_ERROR_THREAD_FAILURE` |

### Custom memory allocators

Define before including `vdb.h`:

```c
#define VDB_MALLOC my_malloc
#define VDB_FREE my_free
#define VDB_REALLOC my_realloc
#include "vdb.h"
```

### Thread safety

When compiled with `VDB_MULTITHREADED`, all operations are thread-safe using read-write locks:

- Multiple threads can search simultaneously
- Add/remove operations are exclusive
- No external locking required

### File format

vdb uses a binary format with magic number `0x56444230`:

- Header: magic (4 bytes), dimensions, count, metric
- Vectors: float array + ID length + ID string (for each vector)
- Metadata is not persisted

### License

Apache v2.0 License
