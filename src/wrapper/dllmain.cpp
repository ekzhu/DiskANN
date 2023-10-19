#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "pch.h"
#include "index.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace po = boost::program_options;

#ifdef _WINDOWS
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
#else
__attribute__((constructor)) void libhnsw_init()
{
    // Code to run when the library is loaded
}

__attribute__((destructor)) void libhnsw_fini()
{
    // Code to run when the library is unloaded
}
#endif

// Global map to store DiskANN indices.
std::unordered_map<uint64_t, std::unique_ptr<diskann::AbstractIndex>> g_index_map;
uint64_t g_next_index_id = 1;

PINVOKELIB_API uint64_t create_in_memory_static_index(size_t dim, size_t max_points, const char *data_type,
                                                      const char *dist_fn, uint32_t num_threads_for_build, uint32_t R,
                                                      uint32_t L, float alpha, uint32_t build_PQ_bytes, bool use_opq)
{
    bool use_pq_build = (build_PQ_bytes > 0);

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::FAST_L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently "
                     "only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                  .with_alpha(alpha)
                                  .with_saturate_graph(false)
                                  .with_num_threads(num_threads_for_build)
                                  .build();
    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(dim)
                      .with_max_points(max_points)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(data_type)
                      .is_dynamic_index(false)
                      .with_index_write_params(index_build_params)
                      .is_enable_tags(true)
                      .is_use_opq(use_opq)
                      .is_pq_dist_build(use_pq_build)
                      .with_num_pq_chunks(build_PQ_bytes)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();

    uint64_t index_id = g_next_index_id++;
    g_index_map[index_id] = std::move(index);

    return index_id;
}

PINVOKELIB_API uint64_t create_in_memory_dynamic_index(std::size_t dim, std::size_t max_points, const char *data_type,
                                                       const char *dist_fn, uint32_t num_threads, uint32_t R,
                                                       uint32_t L, float alpha, uint32_t C, float startPointNorm)
{
    const bool saturate_graph = false;
    bool has_labels = false;

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::FAST_L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently "
                     "only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    auto params = diskann::IndexWriteParametersBuilder(L, R)
                      .with_max_occlusion_size(C)
                      .with_alpha(alpha)
                      .with_num_threads(num_threads)
                      .with_saturate_graph(saturate_graph)
                      .build();
    auto index_search_params = diskann::IndexSearchParams(L, num_threads);
    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(dim)
                      .with_max_points(max_points)
                      .is_dynamic_index(true)
                      .is_enable_tags(true)
                      .is_use_opq(false)
                      .is_filtered(has_labels)
                      .with_num_pq_chunks(0)
                      .is_pq_dist_build(false)
                      .with_tag_type(diskann_type_to_name<uint32_t>())
                      .with_data_type(data_type)
                      .with_index_write_params(params)
                      .with_index_search_params(index_search_params)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .build();
    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->set_start_points_at_random(startPointNorm);

    uint64_t index_id = g_next_index_id++;
    g_index_map[index_id] = std::move(index);
    return index_id;
}

PINVOKELIB_API void build_static(uint64_t index_ptr, float *data, uint32_t *ids, size_t num_points)
{
    auto it = g_index_map.find(index_ptr);
    if (it == g_index_map.end())
    {
        throw diskann::ANNException("Index not found", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    auto index = it->second.get();
    std::vector<uint32_t> tags(num_points);
    for (uint32_t i = 0; i < num_points; i++)
    {
        tags[i] = ids[i];
    }
    index->build(data, num_points, tags);
}

PINVOKELIB_API void load_index(uint64_t index_ptr, const char *index_path, uint32_t num_threads, uint32_t search_l)
{
    auto it = g_index_map.find(index_ptr);
    if (it == g_index_map.end())
    {
        throw diskann::ANNException("Index not found", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    auto index = it->second.get();
    index->load(index_path, num_threads, search_l);
}

PINVOKELIB_API void dispose_index(uint64_t index_ptr)
{
    auto it = g_index_map.find(index_ptr);
    if (it == g_index_map.end())
    {
        throw diskann::ANNException("Index not found", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    // The unique_ptr will automatically delete the index when it goes out of scope
    g_index_map.erase(it);
}

PINVOKELIB_API void save_index(uint64_t index_ptr, const char *index_path)
{
    auto it = g_index_map.find(index_ptr);
    if (it == g_index_map.end())
    {
        throw diskann::ANNException("Index not found", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    it->second->save(index_path);
}

PINVOKELIB_API void insert_point(uint64_t index_ptr, uint32_t point_id, float *point)
{
    auto it = g_index_map.find(index_ptr);
    if (it == g_index_map.end())
    {
        throw diskann::ANNException("Index not found", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    it->second->insert_point(point, point_id);
}

PINVOKELIB_API void query_point(uint64_t index_ptr, float *query, const size_t k, const uint32_t L,
                                uint32_t *result_ids, float *result_dists)
{
    auto it = g_index_map.find(index_ptr);
    if (it == g_index_map.end())
    {
        throw diskann::ANNException("Index not found", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::vector<float *> res_vec;
    it->second->search_with_tags(query, k, L, result_ids, result_dists, res_vec);
}

PINVOKELIB_API void lazy_delete_point(uint64_t index_ptr, uint32_t point_id)
{
    auto it = g_index_map.find(index_ptr);
    if (it == g_index_map.end())
    {
        throw diskann::ANNException("Index not found", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    it->second->lazy_delete(point_id);
}

PINVOKELIB_API void consolidate_deletes(uint64_t index_ptr, uint32_t num_threads, uint32_t L, uint32_t R, float alpha,
                                        uint32_t C)
{
    auto it = g_index_map.find(index_ptr);
    if (it == g_index_map.end())
    {
        throw diskann::ANNException("Index not found", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    diskann::IndexWriteParameters params = diskann::IndexWriteParametersBuilder(L, R)
                                               .with_max_occlusion_size(C)
                                               .with_alpha(alpha)
                                               .with_saturate_graph(false)
                                               .with_num_threads(num_threads)
                                               .build();
    it->second->consolidate_deletes(params);
}