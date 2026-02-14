// Copyright 2026 Abdi Moalim
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef VDB_H
#define VDB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef VDB_MULTITHREADED
#include <pthread.h>
#endif

#ifndef VDB_MALLOC
#define VDB_MALLOC malloc
#endif

#ifndef VDB_FREE
#define VDB_FREE free
#endif

#ifndef VDB_REALLOC
#define VDB_REALLOC realloc
#endif

typedef enum {
  VDB_OK = 0,
  VDB_ERROR_NULL_POINTER = -1,
  VDB_ERROR_INVALID_DIMENSIONS = -2,
  VDB_ERROR_OUT_OF_MEMORY = -3,
  VDB_ERROR_NOT_FOUND = -4,
  VDB_ERROR_INVALID_INDEX = -5,
  VDB_ERROR_THREAD_FAILURE = -6
} vdb_error;

typedef enum {
  VDB_METRIC_COSINE = 0,
  VDB_METRIC_EUCLIDEAN = 1,
  VDB_METRIC_DOT_PRODUCT = 2
} vdb_metric;

typedef struct {
  float* data;
  char* id;
  void* metadata;
} vdb_vector;

typedef struct {
  vdb_vector* vectors;
  size_t count;
  size_t capacity;
  size_t dimensions;
  vdb_metric metric;
#ifdef VDB_MULTITHREADED
  pthread_rwlock_t lock;
#endif
} vdb_database;

typedef struct {
  size_t index;
  float distance;
  char* id;
  void* metadata;
} vdb_result;

typedef struct {
  vdb_result* results;
  size_t count;
} vdb_result_set;

#ifdef VDB_MULTITHREADED
typedef struct {
  const float* query;
  const vdb_database* db;
  vdb_result* results;
  size_t start_idx;
  size_t end_idx;
  size_t k;
} vdb_search_thread_args;
#endif

static inline float vdb_dot_product(const float* a, const float* b,
                                    size_t dims) {
  float sum = 0.0f;

  for (size_t i = 0; i < dims; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

static inline float vdb_magnitude(const float* v, size_t dims) {
  float sum = 0.0f;

  for (size_t i = 0; i < dims; i++) {
    sum += v[i] * v[i];
  }

  return sqrtf(sum);
}

static inline float vdb_cosine_similarity(const float* a, const float* b,
                                          size_t dims) {
  float dot = vdb_dot_product(a, b, dims);
  float mag_a = vdb_magnitude(a, dims);
  float mag_b = vdb_magnitude(b, dims);
  if (mag_a == 0.0f || mag_b == 0.0f)
    return 0.0f;
  return dot / (mag_a * mag_b);
}

static inline float vdb_euclidean_distance(const float* a, const float* b,
                                           size_t dims) {
  float sum = 0.0f;

  for (size_t i = 0; i < dims; i++) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sqrtf(sum);
}

static inline float vdb_compute_distance(const float* a, const float* b,
                                         size_t dims, vdb_metric metric) {
  switch (metric) {
  case VDB_METRIC_COSINE:
    return 1.0f - vdb_cosine_similarity(a, b, dims);
  case VDB_METRIC_EUCLIDEAN:
    return vdb_euclidean_distance(a, b, dims);
  case VDB_METRIC_DOT_PRODUCT:
    return -vdb_dot_product(a, b, dims);
  default:
    return 0.0f;
  }
}

static inline vdb_database* vdb_create(size_t dimensions, vdb_metric metric) {
  if (dimensions == 0)
    return NULL;

  vdb_database* db = (vdb_database*)VDB_MALLOC(sizeof(vdb_database));
  if (!db)
    return NULL;

  db->vectors = NULL;
  db->count = 0;
  db->capacity = 0;
  db->dimensions = dimensions;
  db->metric = metric;

#ifdef VDB_MULTITHREADED
  if (pthread_rwlock_init(&db->lock, NULL) != 0) {
    VDB_FREE(db);
    return NULL;
  }
#endif

  return db;
}

static inline vdb_error vdb_add_vector(vdb_database* db, const float* data,
                                       const char* id, void* metadata) {
  if (!db || !data)
    return VDB_ERROR_NULL_POINTER;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_wrlock(&db->lock);
#endif

  if (db->count >= db->capacity) {
    size_t new_capacity = db->capacity == 0 ? 16 : db->capacity * 2;
    vdb_vector* new_vectors = (vdb_vector*)VDB_REALLOC(
        db->vectors, new_capacity * sizeof(vdb_vector));
    if (!new_vectors) {
#ifdef VDB_MULTITHREADED
      pthread_rwlock_unlock(&db->lock);
#endif
      return VDB_ERROR_OUT_OF_MEMORY;
    }
    db->vectors = new_vectors;
    db->capacity = new_capacity;
  }

  vdb_vector* vec = &db->vectors[db->count];

  vec->data = (float*)VDB_MALLOC(db->dimensions * sizeof(float));
  if (!vec->data) {
#ifdef VDB_MULTITHREADED
    pthread_rwlock_unlock(&db->lock);
#endif
    return VDB_ERROR_OUT_OF_MEMORY;
  }

  memcpy(vec->data, data, db->dimensions * sizeof(float));

  if (id) {
    size_t id_len = strlen(id);
    vec->id = (char*)VDB_MALLOC(id_len + 1);
    if (!vec->id) {
      VDB_FREE(vec->data);
#ifdef VDB_MULTITHREADED
      pthread_rwlock_unlock(&db->lock);
#endif
      return VDB_ERROR_OUT_OF_MEMORY;
    }
    memcpy(vec->id, id, id_len + 1);
  } else {
    vec->id = NULL;
  }

  vec->metadata = metadata;
  db->count++;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_unlock(&db->lock);
#endif

  return VDB_OK;
}

static inline int vdb_result_compare(const void* a, const void* b) {
  const vdb_result* ra = (const vdb_result*)a;
  const vdb_result* rb = (const vdb_result*)b;
  if (ra->distance < rb->distance)
    return -1;
  if (ra->distance > rb->distance)
    return 1;
  return 0;
}

static inline vdb_result_set* vdb_search(const vdb_database* db,
                                         const float* query, size_t k) {
  if (!db || !query || k == 0)
    return NULL;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_rdlock((pthread_rwlock_t*)&db->lock);
#endif

  if (db->count == 0) {
#ifdef VDB_MULTITHREADED
    pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif
    return NULL;
  }

  if (k > db->count)
    k = db->count;

  vdb_result* all_results =
      (vdb_result*)VDB_MALLOC(db->count * sizeof(vdb_result));
  if (!all_results) {
#ifdef VDB_MULTITHREADED
    pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif
    return NULL;
  }

  for (size_t i = 0; i < db->count; i++) {
    all_results[i].index = i;
    all_results[i].distance = vdb_compute_distance(query, db->vectors[i].data,
                                                   db->dimensions, db->metric);
    all_results[i].id = db->vectors[i].id;
    all_results[i].metadata = db->vectors[i].metadata;
  }

  qsort(all_results, db->count, sizeof(vdb_result), vdb_result_compare);

  vdb_result_set* result_set =
      (vdb_result_set*)VDB_MALLOC(sizeof(vdb_result_set));
  if (!result_set) {
    VDB_FREE(all_results);
#ifdef VDB_MULTITHREADED
    pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif
    return NULL;
  }

  result_set->results = (vdb_result*)VDB_MALLOC(k * sizeof(vdb_result));
  if (!result_set->results) {
    VDB_FREE(all_results);
    VDB_FREE(result_set);
#ifdef VDB_MULTITHREADED
    pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif
    return NULL;
  }

  memcpy(result_set->results, all_results, k * sizeof(vdb_result));
  result_set->count = k;

  VDB_FREE(all_results);

#ifdef VDB_MULTITHREADED
  pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif

  return result_set;
}

static inline vdb_error vdb_get_vector(const vdb_database* db, size_t index,
                                       float** out_data, char** out_id,
                                       void** out_metadata) {
  if (!db)
    return VDB_ERROR_NULL_POINTER;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_rdlock((pthread_rwlock_t*)&db->lock);
#endif

  if (index >= db->count) {
#ifdef VDB_MULTITHREADED
    pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif
    return VDB_ERROR_INVALID_INDEX;
  }

  if (out_data)
    *out_data = db->vectors[index].data;
  if (out_id)
    *out_id = db->vectors[index].id;
  if (out_metadata)
    *out_metadata = db->vectors[index].metadata;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif

  return VDB_OK;
}

static inline vdb_error vdb_remove_vector(vdb_database* db, size_t index) {
  if (!db)
    return VDB_ERROR_NULL_POINTER;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_wrlock(&db->lock);
#endif

  if (index >= db->count) {
#ifdef VDB_MULTITHREADED
    pthread_rwlock_unlock(&db->lock);
#endif
    return VDB_ERROR_INVALID_INDEX;
  }

  VDB_FREE(db->vectors[index].data);
  if (db->vectors[index].id) {
    VDB_FREE(db->vectors[index].id);
  }

  if (index < db->count - 1) {
    memmove(&db->vectors[index], &db->vectors[index + 1],
            (db->count - index - 1) * sizeof(vdb_vector));
  }

  db->count--;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_unlock(&db->lock);
#endif

  return VDB_OK;
}

static inline void vdb_free_result_set(vdb_result_set* result_set) {
  if (!result_set)
    return;
  if (result_set->results) {
    VDB_FREE(result_set->results);
  }
  VDB_FREE(result_set);
}

static inline void vdb_destroy(vdb_database* db) {
  if (!db)
    return;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_wrlock(&db->lock);
#endif

  for (size_t i = 0; i < db->count; i++) {
    VDB_FREE(db->vectors[i].data);
    if (db->vectors[i].id) {
      VDB_FREE(db->vectors[i].id);
    }
  }

  if (db->vectors) {
    VDB_FREE(db->vectors);
  }

#ifdef VDB_MULTITHREADED
  pthread_rwlock_unlock(&db->lock);
  pthread_rwlock_destroy(&db->lock);
#endif

  VDB_FREE(db);
}

static inline size_t vdb_count(const vdb_database* db) {
  if (!db)
    return 0;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_rdlock((pthread_rwlock_t*)&db->lock);
  size_t count = db->count;
  pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
  return count;
#else
  return db->count;
#endif
}

static inline size_t vdb_dimensions(const vdb_database* db) {
  if (!db)
    return 0;
  return db->dimensions;
}

static inline vdb_error vdb_save(const vdb_database* db, const char* filename) {
  if (!db || !filename)
    return VDB_ERROR_NULL_POINTER;

#ifdef VDB_MULTITHREADED
  pthread_rwlock_rdlock((pthread_rwlock_t*)&db->lock);
#endif

  FILE* f = fopen(filename, "wb");
  if (!f) {
#ifdef VDB_MULTITHREADED
    pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif
    return VDB_ERROR_OUT_OF_MEMORY;
  }

  uint32_t magic = 0x56444230;
  fwrite(&magic, sizeof(uint32_t), 1, f);

  fwrite(&db->dimensions, sizeof(size_t), 1, f);
  fwrite(&db->count, sizeof(size_t), 1, f);
  fwrite(&db->metric, sizeof(vdb_metric), 1, f);

  for (size_t i = 0; i < db->count; i++) {
    fwrite(db->vectors[i].data, sizeof(float), db->dimensions, f);

    uint32_t id_len =
        db->vectors[i].id ? (uint32_t)strlen(db->vectors[i].id) : 0;
    fwrite(&id_len, sizeof(uint32_t), 1, f);
    if (id_len > 0) {
      fwrite(db->vectors[i].id, sizeof(char), id_len, f);
    }
  }

  fclose(f);

#ifdef VDB_MULTITHREADED
  pthread_rwlock_unlock((pthread_rwlock_t*)&db->lock);
#endif

  return VDB_OK;
}

static inline vdb_database* vdb_load(const char* filename) {
  if (!filename)
    return NULL;

  FILE* f = fopen(filename, "rb");
  if (!f)
    return NULL;

  uint32_t magic;
  if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x56444230) {
    fclose(f);
    return NULL;
  }

  size_t dimensions, count;
  vdb_metric metric;

  if (fread(&dimensions, sizeof(size_t), 1, f) != 1 ||
      fread(&count, sizeof(size_t), 1, f) != 1 ||
      fread(&metric, sizeof(vdb_metric), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  vdb_database* db = vdb_create(dimensions, metric);
  if (!db) {
    fclose(f);
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    float* data = (float*)VDB_MALLOC(dimensions * sizeof(float));
    if (!data || fread(data, sizeof(float), dimensions, f) != dimensions) {
      VDB_FREE(data);
      vdb_destroy(db);
      fclose(f);
      return NULL;
    }

    uint32_t id_len;
    if (fread(&id_len, sizeof(uint32_t), 1, f) != 1) {
      VDB_FREE(data);
      vdb_destroy(db);
      fclose(f);
      return NULL;
    }

    char* id = NULL;
    if (id_len > 0) {
      id = (char*)VDB_MALLOC(id_len + 1);
      if (!id || fread(id, sizeof(char), id_len, f) != id_len) {
        VDB_FREE(data);
        VDB_FREE(id);
        vdb_destroy(db);
        fclose(f);
        return NULL;
      }
      id[id_len] = '\0';
    }

    vdb_error err = vdb_add_vector(db, data, id, NULL);
    VDB_FREE(data);
    VDB_FREE(id);

    if (err != VDB_OK) {
      vdb_destroy(db);
      fclose(f);
      return NULL;
    }
  }

  fclose(f);
  return db;
}

#endif
