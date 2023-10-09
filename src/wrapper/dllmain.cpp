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

PINVOKELIB_API uint64_t create_in_memory_static_index(const char *data_path, const char *index_path_prefix,
                                                      const char *data_type, const char *dist_fn, uint32_t num_threads,
                                                      uint32_t R, uint32_t L, float alpha, uint32_t build_PQ_bytes,
                                                      bool use_opq, const char *label_file, const char *universal_label,
                                                      uint32_t Lf, const char *label_type)
{
    bool use_pq_build;

    use_pq_build = (build_PQ_bytes > 0);

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
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    try
    {
        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;

        size_t data_num, data_dim;
        diskann::get_bin_metadata(data_path, data_num, data_dim);

        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_filter_list_size(Lf)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

        auto filter_params = diskann::IndexFilterParamsBuilder()
                                 .with_universal_label(universal_label)
                                 .with_label_file(label_file)
                                 .with_save_path_prefix(index_path_prefix)
                                 .build();
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(data_dim)
                          .with_max_points(data_num)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type(data_type)
                          .with_label_type(label_type)
                          .is_dynamic_index(false)
                          .with_index_write_params(index_build_params)
                          .is_enable_tags(false)
                          .is_use_opq(use_opq)
                          .is_pq_dist_build(use_pq_build)
                          .with_num_pq_chunks(build_PQ_bytes)
                          .build();

        auto index_factory = diskann::IndexFactory(config);
        auto index = index_factory.create_instance();
        index->build(data_path, data_num, filter_params);
        index->save(index_path_prefix);

        uint64_t index_id = g_next_index_id++;
        g_index_map[index_id] = std::move(index);
        return index_id;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}

PINVOKELIB_API void dispose_index(uint64_t index_ptr)
{
    auto it = g_index_map.find(index_ptr);
    if (it != g_index_map.end())
    {
        // The unique_ptr will automatically delete the index when it goes out of scope
        g_index_map.erase(it);
    }
}

PINVOKELIB_API void query_index(uint64_t index_ptr, float *query, uint32_t k, uint32_t *result_ids, float *result_dists)
{
}