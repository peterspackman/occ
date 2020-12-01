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

class FchkWriter
{
public:
    FchkWriter(const std::string& filename);
    FchkWriter(std::istream&);

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
            m_scalars[key] = std::to_string(value);
        }
    }

    template<typename T>
    void set_vector(const std::string &key, const Eigen::DenseBase<T> &value)
    {
        const auto ptr = value.data();
        if constexpr(std::is_integral<T>::value)
        {
            if constexpr(std::is_same<decltype(value), bool>::value) {
                std::vector<bool> vals;
                vals.reserve(value.size());
                for(size_t i = 0; i < value.size(); i++) {
                    vals.push_back(ptr[i]);
                }
                m_vectors[key] = vals;
            }
            else {
                std::vector<int> vals(value.size());
                vals.reserve(value.size());
                for(size_t i = 0; i < value.size(); i++) {
                    vals.push_back(static_cast<int>(ptr[i]));
                }
                m_vectors[key] = vals;
            }
        }
        else if constexpr(std::is_floating_point<T>::value)
        {
            std::vector<double> vals(value.size());
            vals.reserve(value.size());
            for(size_t i = 0; i < value.size(); i++) {
                vals.push_back(static_cast<int>(ptr[i]));
            }
            m_vectors[key] = vals;
        }
    }
    void write();
private:
    robin_hood::unordered_map<std::string, fchk_scalar> m_scalars;
    robin_hood::unordered_map<std::string, fchk_vector> m_vectors;
};

}


