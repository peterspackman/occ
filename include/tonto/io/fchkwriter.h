#pragma once
#include <istream>
#include <fstream>
#include <vector>
#include <tonto/core/linear_algebra.h>
#include <tonto/qm/basisset.h>
#include <tonto/qm/spinorbital.h>
#include <tonto/3rdparty/robin_hood.h>
#include <variant>
#include <vector>

namespace tonto::io {


namespace impl {
using fchk_vector = std::variant<
    std::vector<double>,
    std::vector<int>,
    std::vector<bool>,
    std::vector<std::string>
>;

using fchk_scalar = std::variant<
    double,
    int,
    bool,
    std::string
>;

struct FchkScalarWriter
{
    std::ostream &destination;
    std::string key;
    void operator()(int);
    void operator()(double);
    void operator()(const std::string&);
    void operator()(bool);
};

struct FchkVectorWriter
{
    std::ostream &destination;
    std::string key;
    void operator()(const std::vector<int>&);
    void operator()(const std::vector<double>&);
    void operator()(const std::vector<std::string>&);
    void operator()(const std::vector<bool>&);
};

}

class FchkWriter
{
public:
    FchkWriter(const std::string& filename);
    FchkWriter(std::ostream&);

    template<typename T>
    void set_scalar(const std::string &key, const T &value)
    {
        if constexpr(std::is_integral<T>::value)
        {
            if constexpr(std::is_same<decltype(value), bool>::value) {
                m_scalars[key] = value;
            }
            else {
                m_scalars[key] = static_cast<int>(value);
            }
        }
        else if constexpr(std::is_floating_point<T>::value)
        {
            m_scalars[key] = static_cast<double>(value);
        }
        else {
            m_scalars[key] = std::string(value);
        }
    }



    template<typename T>
    void set_vector(const std::string &key, const Eigen::DenseBase<T> &mat)
    {
        if constexpr(std::is_integral<typename T::Scalar>::value)
        {
            if constexpr(std::is_same<typename T::Scalar, bool>::value) {
                std::vector<bool> vals;
                vals.reserve(mat.size());
                for(size_t c = 0; c < mat.cols(); c++) {
                    for(size_t r = 0; r < mat.rows(); r++) {
                        vals.push_back(mat(r, c));
                    }
                }
                m_vectors[key] = vals;
            }
            else {
                std::vector<int> vals;
                vals.reserve(mat.size());
                for(size_t c = 0; c < mat.cols(); c++) {
                    for(size_t r = 0; r < mat.rows(); r++) {
                        vals.push_back(static_cast<int>(mat(r, c)));
                    }
                }
                m_vectors[key] = vals;
            }
        }
        else if constexpr(std::is_floating_point<typename T::Scalar>::value)
        {
            std::vector<double> vals;
            vals.reserve(mat.size());
            for(size_t c = 0; c < mat.cols(); c++) {
                for(size_t r = 0; r < mat.rows(); r++) {
                    vals.push_back(static_cast<double>(mat(r, c)));
                }
            }
            m_vectors[key] = vals;
        }
    }
    void write();
private:
    robin_hood::unordered_map<std::string, impl::fchk_scalar> m_scalars;
    robin_hood::unordered_map<std::string, impl::fchk_vector> m_vectors;
    std::ofstream m_owned_destination;
    std::ostream &m_dest;
};

}


