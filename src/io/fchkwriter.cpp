#include <tonto/io/fchkwriter.h>
#include <fmt/ostream.h>

namespace tonto::io {

namespace impl {

static const std::array<const char *, 15> fchk_type_strings {
    "SP", "FOPT", "POPT", "FTS", "FSADDLE", "PSADDLE", "FORCE",
    "FREQ", "SCAN", "GUESS=ONLY", "LST", "STABILITY", "REARCHIVE/MS-RESTART", "MIXED"
};

void FchkScalarWriter::operator()(int value)
{
    fmt::print(destination, "{:40s}   I     {:12d}\n", key, value);
}

void FchkScalarWriter::operator()(double value)
{
    fmt::print(destination, "{:40s}   R     {:22.15e}\n", key, value);
}

void FchkScalarWriter::operator()(const std::string &value)
{
    fmt::print(destination, "{:40s}   C     {:12s}\n", key, value);
}

void FchkScalarWriter::operator()(bool value)
{
    fmt::print(destination, "{:40s}   L     {:1d}\n", key, value);
}


void FchkVectorWriter::operator()(const std::vector<int> &values)
{
    constexpr int num_per_line{6};
    const std::string value_format{"{:12d}"};
    fmt::print(destination, "{:40s}   I    N={:12d}\n", key, values.size());
    int count = 0;
    for(const auto& value: values) {
        fmt::print(destination, value_format, value);
        count++;
        if(count % num_per_line == 0) fmt::print(destination, "\n");
    }
    if(count % num_per_line != 0) fmt::print(destination, "\n");
}

void FchkVectorWriter::operator()(const std::vector<double> &values)
{
    constexpr int num_per_line{5};
    const std::string value_format{"{:16.8e}"};
    fmt::print(destination, "{:40s}   R    N={:12d}\n", key, values.size());
    int count = 0;
    for(const auto& value: values) {
        fmt::print(destination, value_format, value);
        count++;
        if(count % num_per_line == 0) fmt::print(destination, "\n");
    }
    if(count % num_per_line != 0) fmt::print(destination, "\n");
}

void FchkVectorWriter::operator()(const std::vector<std::string> &values)
{
    constexpr int num_per_line{5};
    const std::string value_format{"{:12s}"};
    fmt::print(destination, "{:40s}   C    N={:12d}\n", key, values.size());
    int count = 0;
    for(const auto& value: values) {
        fmt::print(destination, value_format, value);
        count++;
        if(count % num_per_line == 0) fmt::print(destination, "\n");
    }
    if(count % num_per_line != 0) fmt::print(destination, "\n");
}

void FchkVectorWriter::operator()(const std::vector<bool> &values)
{
    constexpr int num_per_line{72};
    const std::string value_format{"{:1d}"};
    fmt::print(destination, "{:40s}   L    N={:12d}\n", key, values.size());
    int count = 0;
    for(const auto& value: values) {
        fmt::print(destination, value_format, value);
        count++;
        if(count % num_per_line == 0) fmt::print(destination, "\n");
    }
    if(count % num_per_line != 0) fmt::print(destination, "\n");
}
}

FchkWriter::FchkWriter(const std::string &filename) : m_owned_destination(filename), m_dest(m_owned_destination)
{

}

FchkWriter::FchkWriter(std::ostream &stream) : m_dest(stream)
{}


void FchkWriter::write()
{
    fmt::print(m_dest, "{:<72s}\n", m_title);
    fmt::print(m_dest, "{:10s}{:30s}{:30s}\n", impl::fchk_type_strings[m_type], m_method, m_basis_name);
    impl::FchkScalarWriter scalar_writer{m_dest, ""};
    for(const auto& keyval: m_scalars)
    {
        scalar_writer.key = keyval.first;
        std::visit(scalar_writer, keyval.second);
    }

    impl::FchkVectorWriter vector_writer{m_dest, ""};
    for(const auto& keyval: m_vectors)
    {
        vector_writer.key = keyval.first;
        std::visit(vector_writer, keyval.second);
    }
}

}
