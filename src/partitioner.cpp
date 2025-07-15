#include "partitioner.h"
#include <omp.h>
#include <stdlib.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  std::string index_file, data_type, gp_file, freq_file;
  unsigned block_size, ldg_times, lock_nums, thead_nums, cut;
  bool use_disk, visual, use_batch;
  std::string reverse_offset_bin, reverse_graph_bin, tmp_edges_filename, sorted_chunks_dir, sorted_reverse_edges_filename;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "print information");
    desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
    desc.add_options()("index_file", po::value<std::string>(&index_file)->required(),
                       "diskann diskann index or mem index");
    desc.add_options()("gp_file", po::value<std::string>(&gp_file)->required(), "output gp file");
    desc.add_options()("reverse_graph_bin", po::value<std::string>(&reverse_graph_bin)->required(),
                       "reverse graph bin file name");
    desc.add_options()("reverse_offset_bin", po::value<std::string>(&reverse_offset_bin)->required(),
                       "reverse graph offset bin file name");
    desc.add_options()("tmp_edges_filename", po::value<std::string>(&tmp_edges_filename)->required(),
                       "tmp reverse edges bin file name for batch graph partition");
    desc.add_options()("sorted_chunks_dir", po::value<std::string>(&sorted_chunks_dir)->required(),
                       "sorted chunks directory for batch graph partition");
    desc.add_options()("sorted_reverse_edges_filename", po::value<std::string>(&sorted_reverse_edges_filename)->required(),
                       "sorted reverse edges bin file name for batch graph partition");                   
    desc.add_options()("freq_file", po::value<std::string>(&freq_file)->default_value(""), "freq_file[optional]");
    desc.add_options()("thread_nums,T", po::value<unsigned>(&thead_nums)->default_value(omp_get_num_procs()),
                       "threads_nums");
    desc.add_options()("lock_nums", po::value<unsigned>(&lock_nums)->default_value(0),
                       "lock node nums, the lock nodes will not participate in the follow LDG paritioning");
    desc.add_options()("block_size,B", po::value<unsigned>(&block_size)->default_value(1),
                       "block size for one partition, 1 for 4KB, 2 for 8KB and so on.");
    desc.add_options()("ldg_times,L", po::value<unsigned>(&ldg_times)->default_value(4),
                       "exec ldg partition alg times, usually 8 is enough.");
    desc.add_options()("use_disk", po::value<bool>(&use_disk)->default_value(1),
                       "Use 1 for use disk index (default), 0 for DiskANN mem index");
    desc.add_options()("visual", po::value<bool>(&visual)->default_value(0),
                       "see real time progress of graph partition");
    desc.add_options()("cut", po::value<unsigned>(&cut)->default_value(INF), "cut adj list, use 3 means graph degree will be cut to 3");
    desc.add_options()("use_batch", po::value<bool>(&use_batch)->default_value(true),
                       "Use batch graph partition, default is true, use false for old version graph partition");


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
  }
  omp_set_num_threads(thead_nums);
  GP::graph_partitioner partitioner(index_file.c_str(), data_type.c_str(), use_disk, block_size, visual,
                                    freq_file, cut, use_batch);
  // partitioner.compute_in_degree<uint8_t>(index_file.c_str());
  // std::cout << "index file: " << index_file << ", data type: " << data_type
  //           << ", gp file: " << gp_file << ", reverse graph bin: " << reverse_graph_bin
  //           << ", reverse offset bin: " << reverse_offset_bin
  //           << ", tmp edges filename: " << tmp_edges_filename
  //           << ", sorted chunks dir: " << sorted_chunks_dir
  //           << ", sorted reverse edges filename: " << sorted_reverse_edges_filename
  //           << std::endl;
  partitioner.batch_write_reverse_index_with_offset<uint8_t>(index_file.c_str(), reverse_graph_bin.c_str(), 
                                                            reverse_offset_bin.c_str(), tmp_edges_filename.c_str(), sorted_chunks_dir.c_str(),
                                                            sorted_reverse_edges_filename.c_str());                      
  // partitioner.validate_reverse_graph<uint8_t>(index_file.c_str(), reverse_graph_bin, reverse_offset_bin);
  // if(use_batch) {
  //   if (std::string(data_type) == std::string("uint8")) {
  //     partitioner.batch_graph_partition<uint8_t>(gp_file.c_str(), ldg_times, index_file.c_str(), reverse_offset_bin, reverse_graph_bin, lock_nums);
  //   } else if (std::string(data_type) == std::string("float")) {
  //     partitioner.batch_graph_partition<float>(gp_file.c_str(), ldg_times, index_file.c_str(), reverse_offset_bin, reverse_graph_bin, lock_nums);
  //   } else {
  //     std::cout << "not support type" << std::endl;
  //     exit(-1);
  //   }
  // }
  // else
  //   partitioner.graph_partition(gp_file.c_str(), ldg_times, lock_nums);
  return 0;
}
