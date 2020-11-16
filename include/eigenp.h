#pragma once

#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <complex>
#include <regex>
#include <assert.h>
#include "robin_hood.h"
#include <cstdio>
#include <zlib.h>
#include <Eigen/Dense>

namespace enpy {

namespace impl {
const std::regex numeric_regex("[0-9]+");
}

inline char host_endian_char()
{
    int x = 1;
    return (reinterpret_cast<char *>(&x))[0] ? '<' : '>';
}

inline char type_char(const std::type_info& t)
{
    // floating types
    if(t == typeid(float)) return 'f';
    if(t == typeid(double)) return 'f';
    if(t == typeid(long double)) return 'f';
    // integral types
    if(t == typeid(int)) return 'i';
    if(t == typeid(char)) return 'i';
    if(t == typeid(short)) return 'i';
    if(t == typeid(long)) return 'i';
    if(t == typeid(long long)) return 'i';
    // unsigned integral types
    if(t == typeid(unsigned int)) return 'u';
    if(t == typeid(unsigned char)) return 'u';
    if(t == typeid(unsigned short)) return 'u';
    if(t == typeid(unsigned long)) return 'u';
    if(t == typeid(unsigned long long)) return 'u';

    if(t == typeid(bool)) return 'b';

    if(t == typeid(std::complex<float>)) return 'c';
    if(t == typeid(std::complex<double>)) return 'c';
    if(t == typeid(std::complex<long double>)) return 'c';
    return '?';
}

struct NumpyArray
{
    NumpyArray(const std::vector<size_t>& shape_, size_t word_size_, bool column_major_)
        : shape(shape_), word_size(word_size_), column_major(column_major_)
    {
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        p_data_buffer = std::make_shared<std::vector<char>>(std::vector<char>(size * word_size));
    }

    NumpyArray() : shape(0), word_size(0), column_major(false), size(0) {}

    template<typename ScalarType>
    ScalarType* data() {
        return reinterpret_cast<ScalarType*>(p_data_buffer->data());
    }

    template<typename ScalarType>
    const ScalarType* data() const
    {
        return reinterpret_cast<ScalarType*>(p_data_buffer->data());
    }

    template<typename ScalarType>
    std::vector<ScalarType> as_std_vector() const
    {
        const ScalarType *p = data<ScalarType>();
        return std::vector<ScalarType>(p, p + size);
    }

    size_t size_bytes() const
    {
        return p_data_buffer->size();
    }

    std::shared_ptr<std::vector<char>> p_data_buffer;
    std::vector<size_t> shape;
    size_t word_size;
    bool column_major{false};
    size_t size;
};

using npz_type = robin_hood::unordered_map<std::string, NumpyArray>;

inline void parse_numpy_header(const std::string header, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order)
{
    size_t location1, location2;

    // read fortran order (column major) flag
    location1 = header.find("fortran_order") + 16;
    if(location1 == std::string::npos)
        throw std::runtime_error("Failed to find header keyword 'fortran_order' in parse_numpy_header");
    fortran_order = header.substr(location1, 4) == "True" ? true : false;

    // read shape
    location1 = header.find("(");
    location2 = header.find(")");
    if (location1 == std::string::npos || location2 == std::string::npos)
        throw std::runtime_error("Failed to find header keyword '(' or ')' in parse_numpy_header");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(location1+1, location2 - location1 + 1);
    while(std::regex_search(str_shape, sm, impl::numeric_regex))
    {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // read endianness, word size, data type
    location1 = header.find("descr") + 9;
    if (location1 == std::string::npos)
        throw std::runtime_error("failed to find header keyword: 'descr' in parse_numpy_header");
    bool little_endian = (header[location1] == '<' || header[location1] == '|') ? true : false;

    // currently don't handle endianness conversion
    assert(little_endian);
    std::string str_ws = header.substr(location1 + 2);
    location2 = str_ws.find("'");
    word_size = std::stoi(str_ws.substr(0, location2));
}

inline void parse_numpy_header(FILE *fp, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order)
{
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if(res != 11) throw std::runtime_error("Failed fread in parse_numpy_header");
    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');
    parse_numpy_header(header, word_size, shape, fortran_order);
}

inline void parse_numpy_header(unsigned char *buffer, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order)
{
    uint8_t major_version = *reinterpret_cast<uint8_t*>(buffer + 6);
    uint8_t minor_version = *reinterpret_cast<uint8_t*>(buffer + 7);
    uint8_t header_length = *reinterpret_cast<uint8_t*>(buffer + 8);
    const std::string header(reinterpret_cast<const char *>(buffer + 9), header_length);
    parse_numpy_header(header, word_size, shape, fortran_order);
}

inline void parse_zip_footer(FILE * fp, uint16_t &num_records, size_t &global_header_size, size_t &global_header_offset)
{
    std::array<char, 22> footer;
    fseek(fp, -22, SEEK_END);
    size_t res = fread(&footer[0], sizeof(char), 22, fp);
    if(res != 22) throw std::runtime_error("Failed fread in parse_zip_footer");
    uint16_t disk_no, disk_start, num_records_on_disk, comment_length;
    disk_no = *(reinterpret_cast<uint16_t *>(&footer[4]));
    disk_start = *(reinterpret_cast<uint16_t *>(&footer[6]));
    num_records_on_disk = *(reinterpret_cast<uint16_t *>(&footer[8]));
    num_records = *(reinterpret_cast<uint16_t *>(&footer[10]));
    global_header_size = *(reinterpret_cast<uint16_t *>(&footer[12]));
    global_header_offset = *(reinterpret_cast<uint16_t *>(&footer[16]));
    comment_length = *(reinterpret_cast<uint16_t *>(&footer[20]));
    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(num_records_on_disk == num_records);
    assert(comment_length == 0);
}

inline NumpyArray parse_npy_array(FILE *fp)
{
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    parse_numpy_header(fp, word_size, shape, fortran_order);

    NumpyArray arr(shape, word_size, fortran_order);
    size_t n = fread(arr.data<char>(), 1, arr.size_bytes(), fp);
    if(n != arr.size_bytes()) throw std::runtime_error("failed fread in parse_npy");
    return arr;
}

inline NumpyArray parse_npz_array(FILE* fp, uint32_t size_compressed, uint32_t size_decompressed)
{

    std::vector<unsigned char> compressed_buffer(size_compressed);
    std::vector<unsigned char> decompressed_buffer(size_decompressed);
    size_t nread = fread(&compressed_buffer[0], 1, size_compressed, fp);
    if(nread != size_compressed)
        throw std::runtime_error("load_the_npy_file: failed fread");

    int err;
    z_stream d_stream;

    d_stream.zalloc = Z_NULL;
    d_stream.zfree = Z_NULL;
    d_stream.opaque = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in = Z_NULL;
    err = inflateInit2(&d_stream, -MAX_WBITS);

    d_stream.avail_in = size_compressed;
    d_stream.next_in = &compressed_buffer[0];
    d_stream.avail_out = size_decompressed;
    d_stream.next_out = &decompressed_buffer[0];

    err = inflate(&d_stream, Z_FINISH);
    err = inflateEnd(&d_stream);

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    parse_numpy_header(&decompressed_buffer[0],word_size,shape,fortran_order);

    NumpyArray array(shape, word_size, fortran_order);

    size_t offset = size_decompressed - array.size_bytes();
    memcpy(array.data<unsigned char>(), &decompressed_buffer[0] + offset, array.size_bytes());
    return array;
}

npz_type load_npz(const std::string &fname) {
    FILE* fp = fopen(fname.c_str(),"rb");

    if(!fp) throw std::runtime_error("Error in load_npz: Unable to open file '"+fname+"'");

    npz_type arrays;

    while(true) {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0],sizeof(char),30,fp);
        if(headerres != 30) throw std::runtime_error("Error in load_npz: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string varname(name_len, ' ');
        size_t vname_res = fread(&varname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len) throw std::runtime_error("Error in load_npz: failed fread");

        //erase the lagging .npy
        varname.erase(varname.end() - 4, varname.end());

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        if(extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res = fread(&buff[0],sizeof(char),extra_field_len,fp);
            if(efield_res != extra_field_len) throw std::runtime_error("Error in load_npz: failed fread");
        }

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0]+8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+22);

        if(compr_method == 0) {arrays[varname] = parse_npy_array(fp);}
        else {arrays[varname] = parse_npz_array(fp,compr_bytes,uncompr_bytes);}
    }
    fclose(fp);

    return arrays;
}

inline NumpyArray load_npz(const std::string &fname, const std::string &varname)
{
    FILE* fp = fopen(fname.c_str(),"rb");

    if(!fp) throw std::runtime_error("Error in load_npz: Unable to open file '"+fname+"'");

    while(1) {
        std::vector<char> local_header(30);
        size_t header_res = fread(&local_header[0],sizeof(char),30,fp);
        if(header_res != 30) throw std::runtime_error("Error in load_npz: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string vname(name_len,' ');
        size_t vname_res = fread(&vname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len) throw std::runtime_error("Error in load_npz: failed fread");
        vname.erase(vname.end()-4,vname.end()); //erase the lagging .npy

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        fseek(fp,extra_field_len,SEEK_CUR); //skip past the extra field

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0]+8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+22);

        if(vname == varname) {
            NumpyArray array  = (compr_method == 0) ? parse_npy_array(fp) : parse_npz_array(fp, compr_bytes, uncompr_bytes);
            fclose(fp);
            return array;
        }
        else {
            //skip past the data
            uint32_t size = *(uint32_t*) &local_header[22];
            fseek(fp,size,SEEK_CUR);
        }
    }

    fclose(fp);

    //if we get here, we haven't found the variable in the file
    throw std::runtime_error("Error in load_npz: Variable name '" + varname + "' not found in '" + fname + "'");
}

inline NumpyArray load_npy(const std::string& fname) {

    FILE* fp = fopen(fname.c_str(), "rb");

    if(!fp) throw std::runtime_error("Error in load_npy: Unable to open file "+fname);
    NumpyArray arr = parse_npy_array(fp);
    fclose(fp);

    return arr;
}

namespace impl {

template<typename T> std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
    //write in little endian
    for(size_t byte = 0; byte < sizeof(T); byte++) {
        char val = *((char*)&rhs+byte);
        lhs.push_back(val);
    }
    return lhs;
}

template<> std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs) {
    lhs.insert(lhs.end(),rhs.begin(),rhs.end());
    return lhs;
}

template<> std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs) {
    //write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for(size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

}

template<typename ScalarType, bool column_major = false>
std::vector<char> create_npy_header(const std::vector<size_t>& shape) {
    using namespace impl;
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += host_endian_char();
    dict += type_char(typeid(ScalarType));
    dict += std::to_string(sizeof(ScalarType));
    dict += "', 'fortran_order': ";
    dict += column_major ? "True" : "False";
    dict += ", 'shape': (";
    dict += std::to_string(shape[0]);

    for(size_t i = 1;i < shape.size();i++) {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if(shape.size() == 1) dict += ",";
    dict += "), }";
    //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(),remainder,' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += (char) 0x93;
    header += "NUMPY";
    header += (char) 0x01; //major version of numpy format
    header += (char) 0x00; //minor version of numpy format
    header += (uint16_t) dict.size();
    header.insert(header.end(),dict.begin(),dict.end());

    return header;
}

template<typename ScalarType, bool column_major = false>
inline void save_npy(const std::string &filename, const ScalarType *data, const std::vector<size_t> &shape, const std::string mode = "w")
{
    FILE *fp = nullptr;
    std::vector<size_t> total_shape;
    if(mode == "a") fp = fopen(filename.c_str(), "r+b");

    if (fp)
    {
        size_t word_size;
        bool fortran_order;
        parse_numpy_header(fp, word_size, total_shape, fortran_order);
        assert(fortran_order == column_major);

        if(word_size != sizeof(ScalarType))
        {
            throw std::runtime_error("Error in save_npy: incompatible word size when appending to array");
        }
        if(total_shape.size() != shape.size())
        {
            throw std::runtime_error("Error in save_npy: incompatible data shape when appending to array");
        }
        for(size_t i = 1; i < shape.size(); i++)
        {
            if(shape[i] != total_shape[i]) {
                throw std::runtime_error("Error in save_npy: incompatible data dimensions when appending");
            }
        }
        total_shape[0] += shape[0];
    }
    else
    {
        fp = fopen(filename.c_str(), "wb");
        total_shape = shape;
    }

    std::vector<char> header = create_npy_header<ScalarType, column_major>(total_shape);
    size_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    fseek(fp, 0, SEEK_SET);
    fwrite(&header[0], sizeof(char), header.size(), fp);
    fseek(fp, 0, SEEK_END);
    fwrite(data, sizeof(ScalarType), num_elements, fp);
    fclose(fp);
}


template<typename ScalarType, bool column_major = false>
void save_npz(const std::string &zipname, const std::string &filename, const ScalarType *data,
              const std::vector<size_t> &shape, std::string mode = "w")
{
    using namespace impl;
    //first, append a .npy to the fname
    const std::string zfname = filename + ".npy";

    //now, on with the show
    FILE* fp = nullptr;
    uint16_t nrecs = 0;
    size_t global_header_offset = 0;
    std::vector<char> global_header;

    if(mode == "a") fp = fopen(zipname.c_str(),"r+b");

    if(fp)
    {
        //zip file exists. we need to add a new npy file to it.
        //first read the footer. this gives us the offset and size of the global header
        //then read and store the global header.
        //below, we will write the the new data at the start of the global header then append the global header and footer below it
        size_t global_header_size;
        parse_zip_footer(fp,nrecs,global_header_size,global_header_offset);
        fseek(fp,global_header_offset,SEEK_SET);
        global_header.resize(global_header_size);
        size_t res = fread(&global_header[0],sizeof(char),global_header_size,fp);
        if(res != global_header_size){
            throw std::runtime_error("Error in save_npz: read error while appending to existing zip");
        }
        fseek(fp,global_header_offset,SEEK_SET);
    }
    else {
        fp = fopen(zipname.c_str(),"wb");
    }

    std::vector<char> npy_header = create_npy_header<ScalarType, column_major>(shape);

    size_t nels = std::accumulate(shape.begin(),shape.end(), 1, std::multiplies<size_t>());
    size_t nbytes = nels*sizeof(ScalarType) + npy_header.size();

    //get the CRC of the data to be added
    uint32_t crc = crc32(0L,(uint8_t*)&npy_header[0],npy_header.size());
    crc = crc32(crc,(uint8_t*)data,nels*sizeof(ScalarType));

    //build the local header
    std::vector<char> local_header;
    local_header += "PK"; //first part of sig
    local_header += (uint16_t) 0x0403; //second part of sig
    local_header += (uint16_t) 20; //min version to extract
    local_header += (uint16_t) 0; //general purpose bit flag
    local_header += (uint16_t) 0; //compression method
    local_header += (uint16_t) 0; //file last mod time
    local_header += (uint16_t) 0;     //file last mod date
    local_header += (uint32_t) crc; //crc
    local_header += (uint32_t) nbytes; //compressed size
    local_header += (uint32_t) nbytes; //uncompressed size
    local_header += (uint16_t) zfname.size(); //fname length
    local_header += (uint16_t) 0; //extra field length
    local_header += zfname;

    //build global header
    global_header += "PK"; //first part of sig
    global_header += (uint16_t) 0x0201; //second part of sig
    global_header += (uint16_t) 20; //version made by
    global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
    global_header += (uint16_t) 0; //file comment length
    global_header += (uint16_t) 0; //disk number where file starts
    global_header += (uint16_t) 0; //internal file attributes
    global_header += (uint32_t) 0; //external file attributes
    global_header += (uint32_t) global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
    global_header += zfname;

    //build footer
    std::vector<char> footer;
    footer += "PK"; //first part of sig
    footer += (uint16_t) 0x0605; //second part of sig
    footer += (uint16_t) 0; //number of this disk
    footer += (uint16_t) 0; //disk where footer starts
    footer += (uint16_t) (nrecs+1); //number of records on this disk
    footer += (uint16_t) (nrecs+1); //total number of records
    footer += (uint32_t) global_header.size(); //nbytes of global headers
    footer += (uint32_t) (global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
    footer += (uint16_t) 0; //zip file comment length

    //write everything
    fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
    fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);
    fwrite(data,sizeof(ScalarType),nels,fp);
    fwrite(&global_header[0],sizeof(char),global_header.size(),fp);
    fwrite(&footer[0],sizeof(char),footer.size(),fp);
    fclose(fp);
}


template<typename T>
void save_npy(const std::string &filename, const Eigen::DenseBase<T> &mat, std::string mode = "w")
{
    std::vector<size_t> shape{
        static_cast<size_t>(mat.rows()), static_cast<size_t>(mat.cols())
    };
    if(mat.IsRowMajor) {
        save_npy<typename T::Scalar, false>(filename, mat.derived().data(), shape, mode);
    }
    else {
        save_npy<typename T::Scalar, true>(filename, mat.derived().data(), shape, mode);
    }
}


template<typename T>
void save_npz(const std::string &zipname, const std::string &filename, const Eigen::DenseBase<T> &mat, std::string mode = "w")
{
    std::vector<size_t> shape{
        static_cast<size_t>(mat.rows()), static_cast<size_t>(mat.cols())
    };
    if(mat.IsRowMajor) {
        save_npz<typename T::Scalar, false>(zipname, filename, mat.derived().data(), shape, mode);
    }
    else {
        save_npz<typename T::Scalar, true>(zipname, filename, mat.derived().data(), shape, mode);
    }
}

template<typename T, bool column_major = false>
void save_npy(const std::string &filename, const std::vector<T> &data, std::string mode = "w")
{
    std::vector<size_t> shape;
    shape.push_back(data.size());
    save_npy<T, column_major>(filename, &data[0], shape, mode);
}

template<typename T, bool column_major = false>
void save_npz(const std::string &zipname, const std::string &fname, const std::vector<T> &data, std::string mode = "w")
{
    std::vector<size_t> shape;
    shape.push_back(data.size());
    save_npz<T, column_major>(zipname, fname, &data[0], shape, mode);
}
}
