//
// Created by Zilliz_wmz on 2022/6/6.
//
#pragma once

#include <omp.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "filesystem"
#include "freq_relayout.h"

#ifndef INF
#define INF 0xffffffff
#endif  // INF
#ifndef READ_U64
#define READ_U64(stream, val) stream.read((char *)&val, sizeof(_u64))
#endif  // !READ_U64
#ifndef ROUND_UP
#define ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y)
#endif  // !ROUND_UP
#ifndef SECTOR_LEN
#define SECTOR_LEN (_u64)4096
#endif  // !SECTOR_LEN

namespace GP {

using concurrent_queue = oneapi::tbb::concurrent_bounded_queue<unsigned>;
namespace fs = std::filesystem;
using _u64 = unsigned long int;
using _u32 = unsigned int;
using VecT = uint8_t;

inline size_t get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    size_t end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    std::cout << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

// DiskANN changes how the meta is stored in the first sector of
// the _disk.index file after commit id 8bb74ff637cb2a77c99b71368ade68c62b7ca8e0
// (exclusive) It returns <is_new_version, vector of metas uint64_ts>
std::pair<bool, std::vector<_u64>> get_disk_index_meta(const std::string &path) {
  std::ifstream fin(path, std::ios::binary);

  int meta_n, meta_dim;
  const int expected_new_meta_n = 9;
  const int expected_new_meta_n_with_reorder_data = 12;
  const int old_meta_n = 11;
  bool is_new_version = true;
  std::vector<_u64> metas;

  fin.read((char *)(&meta_n), sizeof(int));
  fin.read((char *)(&meta_dim), sizeof(int));

  if (meta_n == expected_new_meta_n || meta_n == expected_new_meta_n_with_reorder_data) {
    metas.resize(meta_n);
    fin.read((char *)(metas.data()), sizeof(_u64) * meta_n);
  } else {
    is_new_version = false;
    metas.resize(old_meta_n);
    fin.seekg(0, std::ios::beg);
    fin.read((char *)(metas.data()), sizeof(_u64) * old_meta_n);
  }
  fin.close();
  return {is_new_version, metas};
}

class graph_partitioner {
 public:
  graph_partitioner(const char *indexName, const char *data_type = "uint8",
                    bool load_disk = true, unsigned BS = 1, bool visual = false,
                    std::string freq_file = std::string(""), unsigned cut = INF, bool use_batch = true) {
    _visual = visual;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // check file size
    size_t actual_size = get_file_size(indexName);
    size_t expected_size;
    auto meta_pair = get_disk_index_meta(indexName);
    if (meta_pair.first) {
      expected_size = meta_pair.second.back();
    } else {
      expected_size = meta_pair.second.front();
    }
    if (actual_size != expected_size) {
      std::cout << "index file not match!" << std::endl;
      exit(-1);
    }

    _rd = new std::random_device();
    _gen = new std::mt19937((*_rd)());
    _dis = new std::uniform_real_distribution<>(0, 1);
    if (load_disk) {
      if (std::string(data_type) == std::string("uint8")) {
        if(use_batch) batch_load_disk_index<uint8_t>(indexName, BS);
        else load_disk_index<uint8_t>(indexName, BS);
      } else if (std::string(data_type) == std::string("float")) {
        if(use_batch) batch_load_disk_index<float>(indexName, BS);
        else load_disk_index<float>(indexName, BS);
      } else {
        std::cout << "not support type" << std::endl;
        exit(-1);
      }
    }
    cursize = _nd / 1000;

    if (!freq_file.empty()) {
      if (!fs::exists(freq_file)) {
        std::cout << "No such freq file!" << std::endl;
        exit(-1);
      }
      read_freq(_freq_list, _freq_nei_list, freq_file);
      if (_freq_list.size() != _nd) {
        std::cout << "freq not match, freq file has node " << _freq_list.size() << " but index file nodes is " << _nd
                  << std::endl;
        exit(-1);
      }
      // relayout_adj(_freq_nei_list, full_graph);
    }

    if(!use_batch) {
      // reverse graph
      std::vector<std::mutex> ms(_nd);
      reverse_graph.resize(_nd);
  #pragma omp parallel for shared(reverse_graph, full_graph)
      for (unsigned i = 0; i < _nd; i++) {
        for (unsigned j = 0; j < full_graph[i].size(); j++) {
          std::lock_guard<std::mutex> lock(ms[full_graph[i][j]]);
          reverse_graph[full_graph[i][j]].emplace_back(i);
        }
      }
      std::cout << "reverse graph done." << std::endl;
    }
    

    for (unsigned i = 0; i < _partition_number; i++) {
      if (i % 10000 == 0)
        std::cout << "pmutex pushback " << i << "/" << _partition_number << std::endl;
      pmutex.push_back(std::make_unique<std::mutex>());
    }
  }

  void cout_step() {
    if (!_visual) {
      return;
    }

#pragma omp atomic
    cur++;
    if ((cur + 0) % cursize == 0) {
      std::cout << (double)(cur + 0) / _nd * 100 << "%    \r";
      std::cout.flush();
    }
  }
  
  template <typename T>
  void load_disk_index(const char *index_name, int BS = 1) {
    std::cout << "loading disk index file: " << index_name << "... " << std::flush;
    std::ifstream in;
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
      _u64 expected_npts;
      auto meta_pair = get_disk_index_meta(index_name);

      if (meta_pair.first) {
        // new version
        expected_npts = meta_pair.second.front();
      } else {
        expected_npts = meta_pair.second[1];
      }
      _nd = expected_npts;
      _dim = meta_pair.second[1];

      _max_node_len = meta_pair.second[3];
      C = meta_pair.second[4];

      _partition_number = ROUND_UP(_nd, C) / C;

      std::cout << "_partition_number:" << _partition_number << ",SECTOR_LEN:" << SECTOR_LEN << "max_node_len:" << _max_node_len << std::endl;
      std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(_partition_number * SECTOR_LEN);
      in.open(index_name, std::ios::binary);
      in.seekg(SECTOR_LEN, std::ios::beg);
      in.read(mem_index.get(), _partition_number * SECTOR_LEN);
      in.close();
      full_graph.resize(_nd);
      _u64 des = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
      for (unsigned i = 0; i < _partition_number; i++) {
        if(i % 10000 == 0) 
          std::cout << "partition" << i << "/" << _partition_number << std::endl;
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
        memcpy(sector_buf.get(), mem_index.get() + i * SECTOR_LEN, SECTOR_LEN);
        for (unsigned j = 0; j < C && i * C + j < _nd; j++) {
          if(i * C + j == 65253360) {
            std::cout << "add:" << SECTOR_LEN + i * SECTOR_LEN + j * _max_node_len << std::endl;
          }
          std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
          memcpy(node_buf.get(), sector_buf.get() + j * _max_node_len, _max_node_len);
          unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(T));
          unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(T)) + sizeof(unsigned));
          std::vector<unsigned> tmp(nnbr);
          des += nnbr;
          memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
          full_graph[i * C + j].assign(tmp.begin(), tmp.end());
        }
      }
      std::cout << "avg degree: " << (double)des / _nd << std::endl;
      mem_index.reset();
      C = (SECTOR_LEN * BS) / _max_node_len;
      _partition_number = ROUND_UP(_nd, C) / C;
      std::cout << "_nd: " << _nd << " _dim:" << _dim << " C:" << C << " pn:" << _partition_number << std::endl;
      std::cout << "load index over." << std::endl;
    } catch (std::system_error &e) {
      std::cout << "open file " << index_name << " error!" << std::endl;
      exit(-1);
    }
  }

  template <typename T>
  void compute_in_degree(const char* index_name) {
    _u64 batch_size = 16 * 1024 * 1024 / 4;
    std::ifstream in(index_name, std::ios::binary);
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    auto meta_pair = get_disk_index_meta(index_name);
    _u64 nd = meta_pair.first ? meta_pair.second.front() : meta_pair.second[1];
    _u64 dim = meta_pair.second[1];
    _u64 max_node_len = meta_pair.second[3];
    _u64 C = meta_pair.second[4];
    _u64 partition_number = ROUND_UP(nd, C) / C;

    std::vector<unsigned> in_degree(nd, 0);

    std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(batch_size * SECTOR_LEN);
    in.seekg(SECTOR_LEN, std::ios::beg);
    unsigned batch_num = partition_number / batch_size + 1;

    for (unsigned i = 0; i < batch_num; i++) {
      _u64 current_batch_size = std::min(batch_size, partition_number - i * batch_size);
      in.read(mem_index.get(), current_batch_size * SECTOR_LEN);
      std::cout << "batch " << i << "/" << batch_num << std::endl;

  #pragma omp parallel for schedule(dynamic, 1)
      for (unsigned j = 0; j < current_batch_size; j++) {
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
        memcpy(sector_buf.get(), mem_index.get() + j * SECTOR_LEN, SECTOR_LEN);

        for (unsigned k = 0; k < C && i * batch_size * C + j * C + k < nd; k++) {
          _u64 node_id = i * batch_size * C + j * C + k;
          std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
          memcpy(node_buf.get(), sector_buf.get() + k * max_node_len, max_node_len);

          unsigned& nnbr = *(unsigned*)(node_buf.get() + dim * sizeof(T));
          unsigned* nhood_buf = (unsigned*)(node_buf.get() + dim * sizeof(T) + sizeof(unsigned));

          for (unsigned l = 0; l < nnbr; l++) {
            unsigned dst = nhood_buf[l];
  #pragma omp atomic
            in_degree[dst]++;
          }
        }
      }
    }

    in.close();

    // 输出入度
    std::ofstream deg_out("in_degree.txt");
    for (size_t i = 0; i < in_degree.size(); ++i) {
      deg_out << i << " " << in_degree[i] << "\n";
    }
    deg_out.close();

    std::cout << "Finished writing in-degree to in_degree.txt" << std::endl;
  }

  using Edge = std::pair<unsigned, unsigned>;

  struct EdgeWithFile {
    unsigned dst;
    unsigned src;
    size_t file_index;

    bool operator>(const EdgeWithFile &other) const {
      return dst > other.dst || (dst == other.dst && src > other.src);
    }
  };

  void sort_chunks(const std::string &input_bin_path, size_t chunk_size_bytes, const std::string &chunk_prefix) {
    std::ifstream in(input_bin_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input file");

    size_t edge_size = sizeof(unsigned) * 2;
    size_t edges_per_chunk = chunk_size_bytes / edge_size;

    size_t chunk_id = 0;
    while (!in.eof()) {
      std::vector<Edge> edges;
      edges.reserve(edges_per_chunk);
      unsigned dst, src;

      for (size_t i = 0; i < edges_per_chunk && in.read(reinterpret_cast<char*>(&dst), sizeof(unsigned)); ++i) {
        if (!in.read(reinterpret_cast<char*>(&src), sizeof(unsigned))) break;
        edges.emplace_back(dst, src);
      }

      if (edges.empty()) break;
      std::sort(edges.begin(), edges.end());

      std::ostringstream oss;
      oss << chunk_prefix << "_chunk_" << chunk_id << ".bin";
      std::ofstream chunk_out(oss.str(), std::ios::binary);
      for (const auto &[d, s] : edges) {
        chunk_out.write(reinterpret_cast<const char*>(&d), sizeof(unsigned));
        chunk_out.write(reinterpret_cast<const char*>(&s), sizeof(unsigned));
      }
      std::cout << "Sorted and wrote chunk " << chunk_id << " with " << edges.size() << " edges." << std::endl;
      chunk_id++;
    }
    in.close();
  }

  void merge_sorted_chunks(const std::vector<std::string> &chunk_paths, const std::string &output_path) {
    std::cout << "Merging " << chunk_paths.size() << " sorted chunks..." << std::endl;

    size_t num_files = chunk_paths.size();
    std::vector<std::ifstream> streams(num_files);

    std::priority_queue<EdgeWithFile, std::vector<EdgeWithFile>, std::greater<>> min_heap;

    for (size_t i = 0; i < num_files; ++i) {
      streams[i].open(chunk_paths[i], std::ios::binary);
      unsigned dst, src;
      if (streams[i].read(reinterpret_cast<char*>(&dst), sizeof(unsigned)) &&
          streams[i].read(reinterpret_cast<char*>(&src), sizeof(unsigned))) {
        min_heap.push({dst, src, i});
      }
    }

    std::ofstream out(output_path, std::ios::binary);
    size_t merged = 0;
    while (!min_heap.empty()) {
      EdgeWithFile smallest = min_heap.top();
      min_heap.pop();

      out.write(reinterpret_cast<const char*>(&smallest.dst), sizeof(unsigned));
      out.write(reinterpret_cast<const char*>(&smallest.src), sizeof(unsigned));
      merged++;

      unsigned dst, src;
      size_t idx = smallest.file_index;
      if (streams[idx].read(reinterpret_cast<char*>(&dst), sizeof(unsigned)) &&
          streams[idx].read(reinterpret_cast<char*>(&src), sizeof(unsigned))) {
        min_heap.push({dst, src, idx});
      }
    }
    std::cout << "Merged total of " << merged << " edges." << std::endl;
    for (auto &s : streams) s.close();
    out.close();

    std::cout << "Cleaning up temporary chunk files..." << std::endl;
    for (const auto &path : chunk_paths) {
      if (std::remove(path.c_str()) != 0) {
        std::cerr << "Warning: failed to delete chunk file: " << path << std::endl;
      }
    }
  }

  template <typename T>
  void batch_write_reverse_index_with_offset(const char *index_name,
                                            const char *reverse_output_name,
                                            const char *offset_output_name,
                                            const char *tmp_edges_filename,
                                            const char *sorted_chunks_dir,
                                            const char *sorted_reverse_edges_filename) {
    std::cout << "Reading index metadata..." << std::endl;
    _u64 batch_size = 16 * 1024 * 1024 / 4;
    auto meta_pair = get_disk_index_meta(index_name);
    _u64 nd = meta_pair.first ? meta_pair.second.front() : meta_pair.second[1];
    _u64 dim = meta_pair.second[1];
    _u64 max_node_len = meta_pair.second[3];
    _u64 C = meta_pair.second[4];
    _u64 partition_number = ROUND_UP(nd, C) / C;
    _u64 sector_len = SECTOR_LEN;

    std::ifstream in(index_name, std::ios::binary);
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    in.seekg(sector_len, std::ios::beg);

    std::ofstream tmp_edges_bin(tmp_edges_filename, std::ios::binary);
    if (!tmp_edges_bin) {
      std::cerr << "Failed to open file: " << tmp_edges_filename << std::endl;
      std::exit(1);
    }
    std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(batch_size * sector_len);
    unsigned batch_num = partition_number / batch_size + 1;

    std::cout << "Generating tmp_reverse_edges.bin from original index..." << std::endl;
    for (unsigned i = 0; i < batch_num; i++) {
      _u64 current_batch_size = std::min(batch_size, partition_number - i * batch_size);
      in.read(mem_index.get(), current_batch_size * sector_len);

  #pragma omp parallel for schedule(dynamic)
      for (unsigned j = 0; j < current_batch_size; j++) {
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(sector_len);
        memcpy(sector_buf.get(), mem_index.get() + j * sector_len, sector_len);

        for (unsigned k = 0; k < C && i * batch_size * C + j * C + k < nd; k++) {
          _u64 node_id = i * batch_size * C + j * C + k;
          std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
          memcpy(node_buf.get(), sector_buf.get() + k * max_node_len, max_node_len);

          unsigned &nnbr = *(unsigned *)(node_buf.get() + dim * sizeof(T));
          unsigned *nhood_buf = (unsigned *)(node_buf.get() + (dim * sizeof(T)) + sizeof(unsigned));

  #pragma omp critical
          for (unsigned l = 0; l < nnbr; l++) {
            unsigned dst = nhood_buf[l];
            tmp_edges_bin.write(reinterpret_cast<const char *>(&dst), sizeof(unsigned));
            tmp_edges_bin.write(reinterpret_cast<const char *>(&node_id), sizeof(unsigned));
          }
        }
      }
      std::cout << "Processed batch " << i << "/" << batch_num << std::endl;
    }
    in.close();
    tmp_edges_bin.close();

    std::cout << "Starting external sort..." << std::endl;
    sort_chunks(tmp_edges_filename, 5ULL << 30, sorted_chunks_dir);
    
    if (std::remove(tmp_edges_filename) != 0) {
      std::cerr << "Warning: failed to delete tmp_reverse_edges.bin" << std::endl;
    } else {
      std::cout << "Deleted tmp_reverse_edges.bin after successful sort." << std::endl;
    }

    std::vector<std::string> chunk_paths;
    for (size_t i = 0;; ++i) {
      std::ostringstream oss;
      oss << std::string(sorted_chunks_dir) << "_chunk_" << i << ".bin";
      std::ifstream test(oss.str(), std::ios::binary);
      if (!test) break;
      chunk_paths.push_back(oss.str());
    }
    merge_sorted_chunks(chunk_paths, sorted_reverse_edges_filename);

    std::cout << "Building final reverse index and offset file..." << std::endl;
    std::ifstream sorted_in(sorted_reverse_edges_filename, std::ios::binary);
    std::ofstream reverse_out(reverse_output_name, std::ios::binary);
    std::ofstream offset_out(offset_output_name, std::ios::binary);

    std::vector<uint64_t> offsets(nd, 0);
    uint64_t current_offset = 0;
    unsigned current_dst = 0;
    std::vector<unsigned> neighbors;
    _u64 written_nodes = 0;
    unsigned dst, src;

    while (sorted_in.read(reinterpret_cast<char *>(&dst), sizeof(unsigned)) &&
          sorted_in.read(reinterpret_cast<char *>(&src), sizeof(unsigned))) {
      while (written_nodes < dst) {
        offsets[written_nodes++] = current_offset;
        unsigned degree = 0;
        reverse_out.write(reinterpret_cast<const char *>(&degree), sizeof(unsigned));
        current_offset += sizeof(unsigned);
      }

      if (dst != current_dst) {
        if (!neighbors.empty()) {
          offsets[current_dst] = current_offset;
          unsigned degree = neighbors.size();
          reverse_out.write(reinterpret_cast<const char *>(&degree), sizeof(unsigned));
          reverse_out.write(reinterpret_cast<const char *>(neighbors.data()), degree * sizeof(unsigned));
          current_offset += sizeof(unsigned) + degree * sizeof(unsigned);
          neighbors.clear();
          written_nodes++;
        }
        current_dst = dst;
      }
      neighbors.push_back(src);
    }

    if (!neighbors.empty()) {
      offsets[current_dst] = current_offset;
      unsigned degree = neighbors.size();
      reverse_out.write(reinterpret_cast<const char *>(&degree), sizeof(unsigned));
      reverse_out.write(reinterpret_cast<const char *>(neighbors.data()), degree * sizeof(unsigned));
      current_offset += sizeof(unsigned) + degree * sizeof(unsigned);
      written_nodes++;
    }

    while (written_nodes < nd) {
      offsets[written_nodes++] = current_offset;
      unsigned degree = 0;
      reverse_out.write(reinterpret_cast<const char *>(&degree), sizeof(unsigned));
      current_offset += sizeof(unsigned);
    }

    offset_out.write(reinterpret_cast<const char *>(offsets.data()), nd * sizeof(uint64_t));
    sorted_in.close();
    reverse_out.close();
    offset_out.close();
    std::cout << "Reverse index written to " << reverse_output_name << std::endl;
    std::cout << "Offset index written to " << offset_output_name << std::endl;
  }

  template <typename T>
  void validate_reverse_graph(const std::string& index_path,
                            const std::string& reverse_graph_bin,
                            const std::string& reverse_offset_bin) {
    
    std::ifstream rev_graph_in(reverse_graph_bin, std::ios::binary);
    std::ifstream rev_offset_in(reverse_offset_bin, std::ios::binary);
    std::ifstream index_in(index_path, std::ios::binary);

    if (!rev_graph_in || !rev_offset_in || !index_in) {
      std::cerr << "Failed to open one or more files.\n";
      return;
    }

    // === 获取图元信息 ===
    auto meta_pair = get_disk_index_meta(index_path);
    _u64 nd = meta_pair.first ? meta_pair.second.front() : meta_pair.second[1];
    _u64 dim = meta_pair.second[1];
    _u64 max_node_len = meta_pair.second[3];
    _u64 C = meta_pair.second[4];
    _u64 partition_number = (nd + C - 1) / C;
    _u64 sector_len = SECTOR_LEN;

    // === 加载 offset ===
    std::vector<uint64_t> offsets(nd + 1);
    rev_offset_in.read(reinterpret_cast<char*>(offsets.data()), nd * sizeof(uint64_t));
    offsets[nd] = static_cast<uint64_t>(-1); // sentinel

    index_in.seekg(sector_len, std::ios::beg); // skip metadata

    size_t total = 0, errors = 0;

    for (size_t p = 0; p < partition_number; ++p) {
      if (p % 1000 == 0)
        std::cout << "page: " << p << "/" << partition_number << std::endl;

      std::vector<char> sector_buf(sector_len);
      index_in.read(sector_buf.data(), sector_len);

      for (size_t k = 0; k < C && p * C + k < nd; ++k) {
        size_t node_id = p * C + k;
        char* node_ptr = sector_buf.data() + k * max_node_len;

        uint32_t nnbr = *reinterpret_cast<uint32_t*>(node_ptr + dim * sizeof(T));
        uint32_t* nhood = reinterpret_cast<uint32_t*>(node_ptr + dim * sizeof(T) + sizeof(uint32_t));

        for (uint32_t i = 0; i < nnbr; ++i) {
          uint32_t dst = nhood[i];
          total++;

          uint64_t offset = offsets[dst];
          uint32_t degree;
          rev_graph_in.seekg(offset, std::ios::beg);
          rev_graph_in.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));

          std::vector<uint32_t> rev_nbrs(degree);
          if (degree > 0)
            rev_graph_in.read(reinterpret_cast<char*>(rev_nbrs.data()), degree * sizeof(uint32_t));

          if (std::find(rev_nbrs.begin(), rev_nbrs.end(), node_id) == rev_nbrs.end()) {
            errors++;
            if (errors <= 10) {
              std::cerr << "Missing reverse edge: " << node_id << " → " << dst << std::endl;
            }
          }
        }
      }
    }

    std::cout << "Validation complete. Total edges: " << total
              << ", missing reverse edges: " << errors
              << ", error rate: " << (100.0 * errors / total) << "%" << std::endl;
  }


  template <typename T>
  void batch_load_disk_index(const char *index_name, int BS = 1) {
    _u64 batch_size = 16 * 1024 * 1024 / 4;   //限制64GB内存
    std::cout << "loading disk index file" << index_name << "with batch size" << batch_size << "... " << std::flush;
    std::ifstream in;
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
      _u64 expected_npts;
      auto meta_pair = get_disk_index_meta(index_name);

      if (meta_pair.first) {
        // new version
        expected_npts = meta_pair.second.front();
      } else {
        expected_npts = meta_pair.second[1];
      }
      _nd = expected_npts;
      _dim = meta_pair.second[1];

      _max_node_len = meta_pair.second[3];
      C = meta_pair.second[4];

      _partition_number = ROUND_UP(_nd, C) / C;

      std::cout << "_partition_number:" << _partition_number << ",SECTOR_LEN:" << SECTOR_LEN << std::endl;
      // std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(batch_size * SECTOR_LEN);
      // in.open(index_name, std::ios::binary);
      // in.seekg(SECTOR_LEN, std::ios::beg);
      // full_graph.resize(_nd);

//       unsigned batch_num = _partition_number / batch_size + 1;
//       _u64 des = 0;
//       for(unsigned i = 0; i < batch_num; i++){
//         _u64 current_batch_size = std::min(batch_size, _partition_number - i * batch_size);
//         in.read(mem_index.get(), current_batch_size * SECTOR_LEN);
        
//         std::cout << "batch " << i << "/" << batch_num << std::endl;
// #pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
//         for (unsigned j = 0; j < current_batch_size; j++) {
//           if(j % 10000 == 0) 
//             std::cout << "partition" << j << "/" << current_batch_size << std::endl;
//           std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
//           memcpy(sector_buf.get(), mem_index.get() + j * SECTOR_LEN, SECTOR_LEN);
//           for (unsigned k = 0; k < C && i * batch_size * C + j * C + k < _nd; k++) {
//             std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
//             memcpy(node_buf.get(), sector_buf.get() + k * _max_node_len, _max_node_len);
//             unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(T));
//             unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(T)) + sizeof(unsigned));
//             std::vector<unsigned> tmp(nnbr);
//             des += nnbr;
//             memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
//             full_graph[i * batch_size * C + j * C + k].assign(tmp.begin(), tmp.end());
//           }
//         }
//       }
      
      // in.close();
      
      // std::cout << "avg degree: " << (double)des / _nd << std::endl;
      // mem_index.reset();
      C = (SECTOR_LEN * BS) / _max_node_len;
      _partition_number = ROUND_UP(_nd, C) / C;
      std::cout << "_nd: " << _nd << " _dim:" << _dim << " C:" << C << " pn:" << _partition_number << std::endl;
      // std::cout << "load index over." << std::endl;
    } catch (std::system_error &e) {
      std::cout << "open file " << index_name << " error!" << std::endl;
      exit(-1);
    }
  }

  /**
   * save the partition result
   * @tparam T
   * @param filename
   * @param partition
   */
  void save_partition(const char *filename) {
    // re_id2pid();
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    std::cout << "writing bin: " << filename << std::endl;
    writer.write((char *)&C, sizeof(_u64));
    writer.write((char *)&_partition_number, sizeof(_u64));
    writer.write((char *)&_nd, sizeof(_u64));
    std::cout << "_partition_num: " << _partition_number << " C: " << C << " _nd: " << _nd << std::endl;
    for (unsigned i = 0; i < _partition_number; i++) {
      auto p = _partition[i];
      unsigned s = p.size();
      writer.write((char *)&s, sizeof(unsigned));
      writer.write((char *)p.data(), sizeof(unsigned) * s);
    }
    std::vector<unsigned> id2pidv(_nd);
    for (auto n : id2pid) {
      id2pidv[n.first] = n.second;
    }
    writer.write((char *)id2pidv.data(), sizeof(unsigned) * _nd);
  }

  /**
   * load partition from disk
   * @param filename
   */
  void load_partition(const char *filename) {
    std::ifstream reader(filename, std::ios::binary);
    reader.read((char *)&C, sizeof(_u64));
    reader.read((char *)&_partition_number, sizeof(_u64));
    reader.read((char *)&_nd, sizeof(_u64));
    std::cout << "load partition _partition_num: " << _partition_number << ", C: " << C << std::endl;
    _partition.clear();
    auto tmp = new unsigned[C];
    for (unsigned i = 0; i < _partition_number; i++) {
      unsigned c;
      reader.read((char *)&c, sizeof(unsigned));
      reader.read((char *)tmp, c * sizeof(unsigned));
      std::vector<unsigned> tt;
      tt.reserve(C);
      for (unsigned j = 0; j < c; j++) {
        tt.push_back(*(tmp + j));
      }
      _partition.push_back(tt);
    }
    delete[] tmp;
    re_id2pid();
  }
  void re_id2pid() {
    id2pid.clear();
    for (unsigned i = 0; i < _partition_number; i++) {
      for (unsigned j = 0; j < _partition[i].size(); j++) {
        id2pid[_partition[i][j]] = i;
      }
    }
  }
  /**
   * count the id overlap according to the graph partitioning
   */
  void partition_statistic() {
    std::vector<unsigned> overlap(_nd, 0);
    std::vector<unsigned> blk_neighbor_overlap(_partition_number, 0);
    double overlap_ratio = 0;

#pragma omp parallel for schedule(dynamic, 100) reduction(+ : overlap_ratio)
    for (size_t i = 0; i < _partition_number; i++) {
      std::unordered_set<unsigned> neighbors;
      unsigned blk_neighbor_num = 0;
      for (size_t j = 0; j < _partition[i].size(); j++) {
        blk_neighbor_num += full_graph[_partition[i][j]].size();
        std::unordered_set<unsigned> ne;
        for (unsigned &x : full_graph[_partition[i][j]]) {
          neighbors.insert(x);
          ne.insert(x);
        }
        blk_neighbor_overlap[i] = blk_neighbor_num - neighbors.size();
        for (size_t z = 0; z < _partition[i].size(); z++) {
          if (_partition[i][j] == _partition[i][z]) continue;
          if (ne.find(_partition[i][z]) != ne.end()) {
            overlap[_partition[i][j]]++;
          }
        }
        overlap_ratio +=
            (_partition[i].size() == 1 ? 0 : (1.0 * overlap[_partition[i][j]] / (_partition[i].size() - 1)));
      }
    }
    unsigned max_overlaps = 0;
    unsigned min_overlaps = std::numeric_limits<unsigned>::max();
    double ave_overlap_ratio = 0;
    std::map<unsigned, unsigned> overlap_count;
    for (size_t i = 0; i < _nd; i++) {
      if (overlap_count.count(overlap[i])) {
        overlap_count[overlap[i]]++;
      } else {
        overlap_count[overlap[i]] = 1;
      }
      if (overlap[i] > max_overlaps) max_overlaps = overlap[i];
      if (overlap[i] < min_overlaps) min_overlaps = overlap[i];
    }
    ave_overlap_ratio = overlap_ratio / (double)_nd;
    for (auto &it : overlap_count) {
      std::cout << "each id, overlap number " << it.first << ", count: " << it.second << std::endl;
    }
    std::cout << "each id, max overlaps: " << max_overlaps << std::endl;
    std::cout << "each id, min overlaps: " << min_overlaps << std::endl;
    std::cout << "each id, average overlap ratio: " << ave_overlap_ratio << std::endl;
  }

  unsigned select_partition(unsigned i) {
#pragma omp atomic
    select_nums++;

    float maxn = 0.0;
    unsigned res = INF;
    std::unordered_map<unsigned, unsigned> pcount;
    unsigned tpid = 0;
    // for (auto n : direct_graph[i]) {
    //   unsigned pid = id2pid[n];
    //   if (pid == INF) continue;
    //   pcount[pid] = pcount[pid] + 1;
    //   if (tpid < pid) {
    //     tpid = pid;
    //   }
    // }
    for (auto n : full_graph[i]) {
      unsigned pid = id2pid[n];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    for (auto n : reverse_graph[i]) {
      unsigned pid = id2pid[n];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    // for(unsigned j = 0; j < full_graph.size(); j++){
    //   for(unsigned k = 0; k < full_graph[j].size(); k++){
    //     if(full_graph[j][k] == i) {
    //       unsigned pid = id2pid[j];
    //       if (pid == INF) continue;
    //     pcount[pid] = pcount[pid] + 1;
    //     if (tpid < pid) {
    //       tpid = pid;
    //     }
    //     }
    //   }
    // }
    for (auto c : pcount) {
      unsigned pid = c.first;
      float cnt = c.second;
      std::lock_guard<std::mutex> lock(*pmutex[pid]);
      double s = _partition[pid].size();
      cnt *= (1 - s / C);
      if (cnt > maxn && _partition[pid].size() < C) {
        res = pid;
        maxn = cnt;
      }
    }
    pcount.clear();
    if (res == INF) {
#pragma omp atomic
      select_free++;
      res = getUnfilled();
    }
    select_partition_cnt++;
    return res;
  }

  unsigned batch_select_partition(unsigned i) {
#pragma omp atomic
    select_nums++;

    float maxn = 0.0;
    unsigned res = INF;
    std::unordered_map<unsigned, unsigned> pcount;
    unsigned tpid = 0;
    for (auto m : batch_graph[i]) {
      unsigned pid = id2pid[m];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    for (auto m : batch_reverse_graph[i]) {
      unsigned pid = id2pid[m];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
   
    for (auto c : pcount) {
      unsigned pid = c.first;
      float cnt = c.second;
      std::lock_guard<std::mutex> lock(*pmutex[pid]);
      double s = _partition[pid].size();
      cnt *= (1 - s / C);
      if (cnt > maxn && _partition[pid].size() < C) {
        res = pid;
        maxn = cnt;
      }
    }
    pcount.clear();
    if (res == INF) {
#pragma omp atomic
      select_free++;
      res = getUnfilled();
    }
    return res;
  }

  unsigned getUnfilled() {
#pragma omp atomic
    getUnfilled_nums++;
    unsigned res;
    do {
      free_q.pop(res);
    } while (_partition[res].size() == C);
    return res;
  }

  // graph partition
  void graph_partition(const char *filename, int k, int lock_nums = 0) {
    for (unsigned i = 0; i < _nd; i++) {
      id2pid[i] = INF;
    }
    std::cout << "id2pid init over" << std::endl;
    _partition.clear();
    _partition.resize(_partition_number);
    std::unordered_set<unsigned> vis;
    std::vector<unsigned> init_stream;
    init_stream.reserve(_nd);
    if (!_freq_list.empty()) {
      for (auto p : _freq_list) {
        init_stream.emplace_back(p.first);
      }
    } else {
      init_stream.resize(_nd);
      std::iota(init_stream.begin(), init_stream.end(), 0);
    }
    std::cout << "init_stream over" << std::endl;

    for(int i = 0; i < 10; i++) {
      std::cout << "init_stream[" << i << "]:" << init_stream[i] << " ";
    }
    std::cout << std::endl;

    _lock_nodes.clear();
    _lock_pids.clear();
    _lock_nodes.resize(_nd, false);
    _lock_pids.resize(_partition_number, false);
    unsigned pid = 0;
    vis.clear();
    if (lock_nums) {
      std::cout << "lock first " << lock_nums << " nodes at init stage." << std::endl;
    }
    for (auto i : init_stream) {
      if (vis.count(i)) {
        lock_nums--;
        continue;  // has insert into partition
      }
      if (_partition[pid].size() == C) {
        ++pid;
      }
      vis.insert(i);
      _partition[pid].push_back(i);
      id2pid[i] = pid;
      if (lock_nums > 0) {
        _lock_pids[pid] = true;
      }
      for (unsigned s : full_graph[i]) {
        if(i == 34656077) {
          std::cout << "init stage, full_graph[0]:"; 
          for(int r = 0; r < full_graph[i].size(); r++) {
            std::cout << full_graph[i][r] << " ";
          }
          std::cout << std::endl;
        }
        if (vis.count(s)) continue;
        if (_partition[pid].size() == C) {
          ++pid;
          break;
        }
        // std::cout << "s:" << s << ", pid:" << pid << std::endl;
        _partition[pid].push_back(s);
        id2pid[s] = pid;
        vis.insert(s);
      }
      if (lock_nums) --lock_nums;
    }
    int s = 0;
    for (unsigned i = 0; i < _partition_number; i++) {
      if (!_lock_pids[i]) break;
      for (unsigned s : _partition[i]) {
        _lock_nodes[s] = true;
      }
      s++;
    }
    if (_lock_pids[0]) {
      std::cout << "finally, it locks partition nums: " << s << " locks nodes num: " << s * C << std::endl;
    }

    std::cout << "init over." << std::endl;

    for(int i = 0; i < 10; i++) {
      std::cout << "id2page[" << i << "]:" << id2pid[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < k; i++) {
      select_partition_cnt = 0;
      std::cout << "gp time: " << i << std::endl;
      select_free = 0;
      graph_partition_LDG(i);
      std::cout << "select free: " << (double)select_free / _partition_number << std::endl;
      partition_statistic();
      auto ivf_file_name = std::string(filename) + std::string(".ivf") + std::to_string(i + 1);
      std::cout << "total ivf time: " << ivf_time << std::endl;
      save_partition(ivf_file_name.c_str());
    }
    save_partition(filename);
    std::cout << "select pid nums" << select_nums << " get unfilled partition nums: " << getUnfilled_nums << std::endl;
    std::cout << "total ivf time: " << ivf_time << std::endl;
  }

  template <typename T>
  void load_batch_graph(int batch_no, int batch_size, std::vector<unsigned> stream, 
                        const std::string& index_path, std::vector<uint64_t> offsets, const std::string& reverse_graph_bin) {
    std::cout << "load batch graph start" << std::endl;
    batch_graph.clear();
    batch_reverse_graph.clear();
    // index_in.seekg(SECTOR_LEN, std::ios::beg);

    auto meta_pair = get_disk_index_meta(index_path);
    _u64 nd = meta_pair.first ? meta_pair.second.front() : meta_pair.second[1];
    _u64 dim = meta_pair.second[1];
    _u64 max_node_len = meta_pair.second[3];
    _u64 C = meta_pair.second[4];
    _u64 partition_number = (nd + C - 1) / C;
    _u64 sector_len = SECTOR_LEN;
    if(batch_no == 0) {
      std::cout << "nd: " << nd << ", dim: " << dim << ", max_node_len: " << max_node_len 
                << ", _max_node_len: " << _max_node_len
                << ", C: " << C << ", partition_number: " << partition_number 
                << ", sector_len: " << sector_len << std::endl;
    }

    int cur_batch_size = (batch_size > nd - batch_no * batch_size) ? nd - batch_no * batch_size : batch_size;
    // std::cout << "batch_size:" << batch_size << std::endl; 
    // std::cout << "cur_batch_size:" << cur_batch_size << std::endl; 
    batch_graph.resize(cur_batch_size);
    batch_reverse_graph.resize(cur_batch_size);
    // std::cout << "batch graph resize done." << std::endl;
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < cur_batch_size; i++) {
      std::ifstream rev_graph_in(reverse_graph_bin, std::ios::binary);
      std::ifstream index_in(index_path, std::ios::binary);
      // std::cout << "i:" << i << std::endl;
      // std::cout << "stream size:" << stream.size() << std::endl;
      // std::cout << "batch_no * batch_size + i:" << batch_no * batch_size + i << std::endl;
      unsigned node_id = stream[batch_no * batch_size + i];
      unsigned page_id = node_id / C;
      if(batch_no == 0 && i == 0)
        std::cout << "node_id:" << node_id << ", page_id:" << page_id << std::endl;
      // std::cout << "node id:" << node_id << ", page id:" << page_id << std::endl;
      uint64_t rev_graph_offset = offsets[node_id];
      // std::cout << "rev_graph_offset:" << rev_graph_offset << std::endl;
      uint32_t degree;
      rev_graph_in.seekg(rev_graph_offset, std::ios::beg);
      rev_graph_in.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));
      // std::cout << "degree:" << degree << std::endl;
      std::vector<uint32_t> rev_nbrs(degree);
      if (degree > 0)
        rev_graph_in.read(reinterpret_cast<char*>(rev_nbrs.data()), degree * sizeof(uint32_t));
      batch_reverse_graph[i] = rev_nbrs;
      // std::cout << "batch reverse graph done." << std::endl;

      std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
      index_in.seekg(SECTOR_LEN + page_id * sector_len, std::ios::beg);
      index_in.read(sector_buf.get(), SECTOR_LEN);
      
      std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
      memcpy(node_buf.get(), sector_buf.get() + (node_id % C) * _max_node_len, _max_node_len);

      if(node_id == 65253360)
        std::cout << "add: " << SECTOR_LEN + page_id * sector_len + (node_id % C) * max_node_len << std::endl;

      unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(T));
      unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(T)) + sizeof(unsigned));
      std::vector<unsigned> tmp(nnbr);
      // std::cout << "nnbr:" << nnbr << std::endl;
      memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
      for (uint32_t j = 0; j < nnbr; ++j) {
        uint32_t nbr = tmp[j];
        // batch_graph[i].emplace_back(nbr);
        if(node_id == 65253360) std::cout << nbr << " ";
      }
      batch_graph[i].assign(tmp.begin(), tmp.end());
      // std::cout << "batch graph done." << std::endl;
      index_in.close();
      rev_graph_in.close();
    }
    std::cout << "load batch graph done." << "batch graph size:" << batch_graph.size() 
              << ", batch reverse graph size:" << batch_reverse_graph.size() << std::endl;
  }

  // graph partition
  template <typename T>
  void batch_graph_partition(const char *filename, int k, const std::string& index_path,
                              const std::string& reverse_offset_bin, const std::string& reverse_graph_bin, int lock_nums = 0) {
    unsigned batch_size = 100000000;
    unsigned batch_num = (_nd + batch_size - 1) / batch_size;

    std::ifstream rev_offset_in(reverse_offset_bin, std::ios::binary);
    std::vector<uint64_t> offsets(_nd + 1);
    rev_offset_in.read(reinterpret_cast<char*>(offsets.data()), _nd * sizeof(uint64_t));
    offsets[_nd] = static_cast<uint64_t>(-1); // sentinel
    rev_offset_in.close();

    for (unsigned i = 0; i < _nd; i++) {
      id2pid[i] = INF;
    }
    std::cout << "id2pid init over" << std::endl;

    _partition.clear();
    _partition.resize(_partition_number);
    std::unordered_set<unsigned> vis;
    std::vector<unsigned> init_stream;
    init_stream.reserve(_nd);
    if (!_freq_list.empty()) {
      std::cout << "use freq list" << std::endl;
      for (auto p : _freq_list) {
        init_stream.emplace_back(p.first);
      }
    } else {
      init_stream.resize(_nd);
      std::iota(init_stream.begin(), init_stream.end(), 0);
    }

    std::cout << "init_stream over" << ", init_stream size:" << init_stream.size() << std::endl;
    _freq_list.clear();
    _freq_nei_list.clear();
    // for(int i = 0; i < 10; i++) {
    //   std::cout << "init_stream[" << i << "]:" << init_stream[i] << " ";
    // }
    // std::cout << std::endl;

    _lock_nodes.clear();
    _lock_pids.clear();
    _lock_nodes.resize(_nd, false);
    _lock_pids.resize(_partition_number, false);
    unsigned pid = 0;
    vis.clear();
    // full_graph.resize(_nd);

    if (lock_nums) {
      std::cout << "lock first " << lock_nums << " nodes at init stage." << std::endl;
    }
    int iteration = 0;
    for (auto i : init_stream) {
      // std::cout << "A" << std::endl;
      if(iteration % batch_size == 0) 
        load_batch_graph<T>(iteration / batch_size, batch_size, init_stream, index_path, offsets, reverse_graph_bin);

      if (vis.count(i)) {
        lock_nums--;
        iteration++;
        continue;  // has insert into partition
      }
      if (_partition[pid].size() == C) {
        ++pid;
      }
      vis.insert(i);
      _partition[pid].push_back(i);
      id2pid[i] = pid;
      if (lock_nums > 0) {
        _lock_pids[pid] = true;
      }
      
      // std::cout << "full graph i size:" << full_graph[i].size() << std::endl;

      for (unsigned s : batch_graph[iteration % batch_size]) {
        // if(iteration == 0) {
        //   std::cout << "init stage, batch_graph[0]:";
        //   for(int r = 0; r < batch_graph[iteration % batch_size].size(); r++) {
        //     std::cout << batch_graph[iteration % batch_size][r] << " ";
        //   }
        //   std::cout << std::endl;
        // }
        if (vis.count(s)) continue;
        if (_partition[pid].size() == C) {
          ++pid;
          break;
        }
        // std::cout << "s:" << s << ", pid:" << pid << std::endl;
        _partition[pid].push_back(s);
        id2pid[s] = pid;
        vis.insert(s);
      }
      if (lock_nums) --lock_nums;
      iteration++;
    }
    int s = 0;
    for (unsigned i = 0; i < _partition_number; i++) {
      if (!_lock_pids[i]) break;
      for (unsigned s : _partition[i]) {
        _lock_nodes[s] = true;
      }
      s++;
    }
    if (_lock_pids[0]) {
      std::cout << "finally, it locks partition nums: " << s << " locks nodes num: " << s * C << std::endl;
    }

    std::cout << "init over." << std::endl;

    // for(int i = 0; i < 10; i++) {
    //   std::cout << "id2page[" << i << "]:" << id2pid[i] << " ";
    // }
    // std::cout << std::endl;

    // print_memory_breakdown();
    for (int i = 0; i < k; i++) {
      std::cout << "gp time: " << i << std::endl;
      free_q.clear();
#pragma omp parallel for
      for (unsigned i = 0; i < _partition_number; i++) {
        // if (_lock_pids[i]) continue;
        _partition[i].clear();
        free_q.push(i);
      }
      cur = 0;
      std::cout << "start" << std::endl;
      std::vector<unsigned> stream(_nd);
      std::iota(stream.begin(), stream.end(), 0);

      // 使用固定种子初始化随机引擎
      std::default_random_engine rng(42 + i);  // 42 是一个示例，你可以用任何固定的整数
      std::shuffle(stream.begin(), stream.end(), rng);
      // std::cout << "stream: ";
      // for(int j = 0; j < 10; j++) {
      //   std::cout << stream[j] << " ";
      // }
      // std::cout << std::endl;

      auto start = omp_get_wtime();
      // int* loc = new int[_nd];  // 动态分配
      // std::vector<std::mutex> ms(_nd);
      for(int j = 0; j < batch_num; j++){
        std::cout << "batch " << j << " start" << std::endl;
        load_batch_graph<T>(j, batch_size, stream, index_path, offsets, reverse_graph_bin);
        // if(j == 0) {
        //   for(int r = 0; r < 5; r++) {
        //     std::cout << "batch_graph " << r << "(" << stream[j * batch_size + r] << "):" ;
        //     for(int t = 0; t < batch_graph[r].size(); t++)
        //       std::cout << batch_graph[r][t] << " ";
        //     std::cout << std::endl;
        //   }
        //   for(int r = 0; r < 5; r++) {
        //     std::cout << "batch_reverse_graph " << r << "(" << stream[j * batch_size + r] << "):" ;
        //     for(int t = 0; t < batch_reverse_graph[r].size(); t++)
        //       std::cout << batch_reverse_graph[r][t] << " ";
        //     std::cout << std::endl;
        //   }
        // }
        batch_graph_partition_LDG(j, batch_size, stream);
      }
      auto end = omp_get_wtime();
      std::cout << "ivf time: " << end - start << " round: " << round << std::endl;
      ivf_time += end - start;
      round++;

      select_free = 0;
      // graph_partition_LDG();
      std::cout << "select free: " << (double)select_free / _partition_number << std::endl;
      // partition_statistic();
      auto ivf_file_name = std::string(filename) + std::string(".ivf") + std::to_string(i + 1);
      std::cout << "total ivf time: " << ivf_time << std::endl;
      save_partition(ivf_file_name.c_str());
      // for(int i = 0; i < 10; i++) {
      //   std::cout << "id2page[" << i << "]:" << id2pid[i] << " ";
      // }
      // std::cout << std::endl;
    }
    save_partition(filename);
    std::cout << "select pid nums" << select_nums << " get unfilled partition nums: " << getUnfilled_nums << std::endl;
    std::cout << "total ivf time: " << ivf_time << std::endl;
  }

  void graph_partition_LDG(int j) {
    free_q.clear();
#pragma omp parallel for
    for (unsigned i = 0; i < _partition_number; i++) {
      if (_lock_pids[i]) continue;
      _partition[i].clear();
      free_q.push(i);
    }

    cur = 0;
    std::cout << "start" << std::endl;
    std::vector<unsigned> stream(_nd);
    std::iota(stream.begin(), stream.end(), 0);
    std::default_random_engine rng(42 + j);  // 42 是一个示例，你可以用任何固定的整数
    std::shuffle(std::begin(stream), std::end(stream), rng);

    std::cout << "stream: ";
    for(int j = 0; j < 10; j++) {
      std::cout << stream[j] << " ";
    }
    std::cout << std::endl;

    auto start = omp_get_wtime();
    for(int r = 0; r < 5; r++) {
      std::cout << "full_graph " << r << "(" << stream[r] << "):" ;
      for(int t = 0; t < full_graph[stream[r]].size(); t++)
        std::cout << full_graph[stream[r]][t] << " ";
      std::cout << std::endl;
    }
    for(int r = 0; r < 5; r++) {
      std::cout << "reverse_graph " << r << "(" << stream[r] << "):" ;
      for(int t = 0; t < reverse_graph[stream[r]].size(); t++)
        std::cout << reverse_graph[stream[r]][t] << " ";
      std::cout << std::endl;
    }

#pragma omp parallel for schedule(dynamic)
    for (unsigned i = 0; i < _nd; i++) {
      if(i % 10000 == 0) std::cout << "sync " << i << "/" << _nd << std::endl;
      size_t n = stream[i];
      // std::cout << n << " ";
      if (_lock_nodes[n]) continue;
      unsigned pid = sync(n);
      // std::cout << pid << ", ";
      cout_step();
    }
    auto end = omp_get_wtime();
    std::cout << "ivf time: " << end - start << " round: " << round << std::endl;
    ivf_time += end - start;
    round++;

    for(int i = 0; i < 10; i++) {
      std::cout << "id2page[" << i << "]:" << id2pid[i] << " ";
    }
    std::cout << std::endl;
  }

  void batch_graph_partition_LDG(int batch_i, int batch_size, std::vector<unsigned> &stream) {
#pragma omp parallel for schedule(dynamic)
    for (unsigned i = 0; i < batch_size; i++) {
      if(batch_i * batch_size + i >= _nd) continue;
      if((batch_i * batch_size + i) % 100000 == 0) std::cout << "sync " << batch_i * batch_size + i << "/" << _nd << std::endl;
      size_t n = stream[batch_i * batch_size + i];
      // std::cout << n << " ";
      // if (_lock_nodes[n]) continue;
      unsigned pid = batch_sync(n, i);
      // std::cout << pid << ", ";
      cout_step();
    }
  }

  unsigned sync(unsigned i) {
    unsigned pid = select_partition(i);
    pmutex[pid]->lock();

    while (_partition[pid].size() == C) {
      pmutex[pid]->unlock();
      pid = select_partition(i);
      pmutex[pid]->lock();
    }
    _partition[pid].emplace_back(i);
    id2pid[i] = pid;
    unsigned s = _partition[pid].size();
    pmutex[pid]->unlock();

    if (s != C) {
      free_q.push(pid);
    }

    return pid;
  }

  unsigned batch_sync(unsigned n, unsigned i) {
    unsigned pid = batch_select_partition(i);
    pmutex[pid]->lock();

    while (_partition[pid].size() == C) {
      pmutex[pid]->unlock();
      pid = batch_select_partition(i);
      pmutex[pid]->lock();
    }
    _partition[pid].emplace_back(n);
    id2pid[n] = pid;
    unsigned s = _partition[pid].size();
    pmutex[pid]->unlock();

    if (s != C) {
      free_q.push(pid);
    }

    return pid;
  }

 private:
  size_t _dim;  // vector dimension
  _u64 _nd;     // vector number
  _u64 _max_node_len;
  unsigned _width;                                  // max out-degree
  unsigned _ep;                                     // seed vertex id
  int select_partition_cnt = 0;
  std::vector<std::vector<unsigned>> direct_graph;  // neighbor list
  std::vector<std::vector<unsigned>> full_graph;
  std::vector<std::vector<unsigned>> batch_graph;
  std::vector<std::vector<unsigned>> batch_reverse_graph;
  unsigned select_free;
  _u64 C;                                                  // partition size threshold
  _u64 _partition_number = 0;                              // the number of partitions
  std::vector<std::vector<unsigned>> _partition{1000000};  // each partition set
  std::vector<std::unique_ptr<std::mutex>> pmutex;
  int cur = 0;
  std::vector<std::vector<unsigned>> reverse_graph;
  std::vector<std::vector<unsigned>> undirect_graph;
  std::unordered_map<unsigned, unsigned> id2pid;
  std::unordered_map<unsigned, unsigned> id2ratio;
  int round = 0;
  double ivf_time = 0.0;
  bool _visual = false;
  unsigned cursize = 10000;
  uint64_t select_nums = 0;
  uint64_t getUnfilled_nums = 0;
  _u64 E;
  std::uniform_real_distribution<> *_dis;
  std::mt19937 *_gen;
  std::random_device *_rd;
  concurrent_queue free_q;

  std::vector<puu> _freq_list;
  vpu _freq_nei_list;
  std::vector<bool> _lock_nodes;
  std::vector<bool> _lock_pids;
};
}  // namespace GP