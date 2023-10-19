#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define PINVOKELIB_API __declspec(dllexport)
#else
#define PINVOKELIB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    PINVOKELIB_API uint64_t create_in_memory_static_index(size_t dim, size_t max_points, const char *data_type,
                                                          const char *dist_fn, uint32_t num_threads, uint32_t R,
                                                          uint32_t L, float alpha, uint32_t build_PQ_bytes,
                                                          bool use_opq);
    PINVOKELIB_API uint64_t create_in_memory_dynamic_index(size_t dim, size_t max_points, const char *data_type,
                                                           const char *dist_fn, uint32_t R, uint32_t L, float alpha,
                                                           uint32_t C);
    PINVOKELIB_API void build_static(uint64_t index_ptr, float *data, uint32_t *ids, size_t num_points);
    PINVOKELIB_API void dispose_index(uint64_t index_ptr);
    PINVOKELIB_API void load_index(uint64_t index_ptr, const char *index_path, uint32_t num_threads, uint32_t search_l);
    PINVOKELIB_API void save_index(uint64_t index_ptr, const char *index_path);
    PINVOKELIB_API void insert_point(uint64_t index_ptr, uint32_t point_id, float *point);
    PINVOKELIB_API void lazy_delete_point(uint64_t index_ptr, uint32_t point_id);
    PINVOKELIB_API void consolidate_deletes(uint64_t index_ptr, uint32_t num_threads, uint32_t L, uint32_t R,
                                            float alpha, uint32_t C);
    PINVOKELIB_API void query_point(uint64_t index_ptr, float *query, size_t k, uint32_t L, uint32_t *result_ids,
                                    float *result_dists);

#ifdef __cplusplus
}
#endif