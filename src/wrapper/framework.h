#pragma once

#include <cstdint>

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

    PINVOKELIB_API uint64_t create_in_memory_static_index(const char *data_path, const char *index_path_prefix,
                                                          const char *data_type, const char *dist_fn,
                                                          uint32_t num_threads, uint32_t R, uint32_t L, float alpha,
                                                          uint32_t build_PQ_bytes, bool use_opq,
                                                          const char *label_file = "", const char *universal_label = "",
                                                          uint32_t Lf = 0, const char *label_type = "uint");
    PINVOKELIB_API void dispose_index(uint64_t index_ptr);
    PINVOKELIB_API void query_index(uint64_t index_ptr, float *query, uint32_t k, uint32_t *result_ids,
                                    float *result_dists);

#ifdef __cplusplus
}
#endif