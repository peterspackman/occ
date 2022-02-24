#pragma once

#include <type_traits>
#include <string>
#include <iterator>
#include <Eigen/Core>
#include <fmt/format.h>

#ifdef EIGEN_FMT_USE_IOFORMAT
#include <sstream>
#endif

namespace EigenFmt {

//! \brief Formatting parameters for Eigen matrices
//!
struct FormatSpec {
    int precision{Eigen::StreamPrecision};
    bool dont_align_cols{false};
    bool transpose{false};
    std::string coeff_sep{" "};
    std::string row_sep{"\n"};
    std::string row_prefix{""};
    std::string row_suffix{""};
    std::string mat_prefix{""};
    std::string mat_suffix{""};

    std::string str() const {
        std::string format_str =
            fmt::format("p{{{}}};csep{{{}}};rsep{{{}}};rpre{{{}}};"
                        "rsuf{{{}}};mpre{{{}}};msuf{{{}}}",
                        precision, coeff_sep, row_sep, row_prefix, row_suffix,
                        mat_prefix, mat_suffix);
        if (transpose) {
            format_str += ";t";
        }
        if (dont_align_cols) {
            format_str += ";noal";
        }
        return format_str;
    }
};

class Format {
public:
    Format& precision(int precision) {
        format_.precision = precision;
        return *this;
    }

    Format& alignCols() {
        format_.dont_align_cols = false;
        return *this;
    }

    Format& dontAlignCols() {
        format_.dont_align_cols = true;
        return *this;
    }

    Format& transpose() {
        format_.transpose = true;
        return *this;
    }

    Format& dontTranspose() {
        format_.transpose = false;
        return *this;
    }

    Format& coeffSep(const std::string& coeff_sep) {
        format_.coeff_sep = coeff_sep;
        return *this;
    }

    Format& rowSep(const std::string& row_sep) {
        format_.row_sep = row_sep;
        return *this;
    }

    Format& rowPrefix(const std::string& row_prefix) {
        format_.row_prefix = row_prefix;
        return *this;
    }

    Format& rowSuffix(const std::string& row_suffix) {
        format_.row_suffix = row_suffix;
        return *this;
    }

    Format& matPrefix(const std::string& mat_prefix) {
        format_.mat_prefix = mat_prefix;
        return *this;
    }

    Format& matSuffix(const std::string& mat_suffix) {
        format_.mat_suffix = mat_suffix;
        return *this;
    }

    std::string str() const {
        return format_.str();
    }

    FormatSpec& spec() {
        return format_;
    }

    const FormatSpec& spec() const {
        return format_;
    }

    operator FormatSpec&() {
        return format_;
    }

private:
    FormatSpec format_;
};

inline Format format() {
    return Format{};
}

namespace detail {
template <typename Derived>
struct is_matrix_expression
    : std::is_base_of<Eigen::DenseBase<typename std::decay<Derived>::type>,
                      typename std::decay<Derived>::type> {};

template <typename T>
std::string format(const T& matrix, const FormatSpec& fmt) {
    static_assert(EigenFmt::detail::is_matrix_expression<T>::value,
                  "EigenFmt::format() can only format Eigen matrices");

    auto out = fmt::memory_buffer();
    if (matrix.size() == 0) {
        format_to(std::back_inserter(out), "{}{}", fmt.mat_prefix,
                  fmt.mat_suffix);
        return {out.data(), out.size()};
    }

    using Scalar = typename T::Scalar;

    Eigen::Index width = 0;

    std::ptrdiff_t explicit_precision;
    if (fmt.precision == Eigen::StreamPrecision) {
        explicit_precision = 0;
    } else if (fmt.precision == Eigen::FullPrecision) {
        if (Eigen::NumTraits<Scalar>::IsInteger) {
            explicit_precision = 0;
        } else {
            explicit_precision =
                Eigen::internal::significant_decimals_impl<Scalar>::run();
        }
    } else {
        explicit_precision = fmt.precision;
    }

    std::string row_spacer;
    if (not fmt.dont_align_cols) {
        int i = int(fmt.mat_suffix.length()) - 1;
        while (i >= 0 && fmt.mat_suffix[static_cast<size_t>(i)] != '\n') {
            row_spacer += ' ';
            i--;
        }
    }

    auto print_coeff = [&width, &out, &explicit_precision](const Scalar& v) {
        if (width > 0) {
            if (explicit_precision != 0) {
                format_to(std::back_inserter(out), "{:{}.{}}", v, width,
                          explicit_precision);
            } else {
                format_to(std::back_inserter(out), "{:{}}", v, width);
            }
        } else {
            if (explicit_precision != 0) {
                format_to(std::back_inserter(out), "{:.{}}", v,
                          explicit_precision);
            } else {
                format_to(std::back_inserter(out), "{}", v);
            }
        }
    };

    auto print_string = [&out](const std::string& str) {
        format_to(std::back_inserter(out), "{}", str);
    };

    if (not fmt.dont_align_cols) {
        // compute the largest width
        for (Eigen::Index j = 0; j < matrix.cols(); ++j)
            for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
                auto str = fmt::format("{}", matrix.coeff(i, j));
                width = std::max<Eigen::Index>(width, str.size());
            }
    }
    print_string(fmt.mat_prefix);
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        if (i) {
            print_string(row_spacer);
        }
        print_string(fmt.row_prefix);
        print_coeff(matrix.coeff(i, 0));
        for (Eigen::Index j = 1; j < matrix.cols(); ++j) {
            print_string(fmt.coeff_sep);
            print_coeff(matrix.coeff(i, j));
        }
        print_string(fmt.row_suffix);
        if (i < matrix.rows() - 1) {
            print_string(fmt.row_sep);
        }
    }
    print_string(fmt.mat_suffix);
    return {out.data(), out.size()};
}

} // namespace detail

template <typename T>
std::string format(const T& matrix, const FormatSpec& fmt) {
    if (fmt.transpose) {
        return detail::format(matrix.transpose(), fmt);
    } else {
        return detail::format(matrix, fmt);
    }
}

template <typename T>
std::string format(const T& matrix, const Format& fmt) {
    const auto& spec = fmt.spec();
    if (spec.transpose) {
        return detail::format(matrix.transpose(), spec);
    } else {
        return detail::format(matrix, spec);
    }
}

}; // namespace EigenFmt

namespace fmt {

template <typename T>
struct formatter<
    T, typename std::enable_if<EigenFmt::detail::is_matrix_expression<T>::value,
                               char>::type> {

    template <typename ParseContext>
    auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin();
        auto end = ctx.end();
        return internal_parse(it, end, &ctx);
    }

    // Needed for internal_parse to be callable from outside parse()
    struct DummyContext {
        int next_arg_id() {
            return 0;
        }
    };

    //! \brief Parses the given iterator range and updates the internal state
    //!
    //! \param it Iterator pointing at the begining of the range to parse
    //! \param end Iterator pointing at the end of the range to parse
    //! \param ctx The parsing context (nullptr if called from outside the
    //! parse() function)
    //! \return Iterator The iterator pointer after the last character parsed
    template <typename Iterator, typename ParseContext = DummyContext>
    Iterator internal_parse(Iterator& it, const Iterator& end,
                            ParseContext* ctx = nullptr) {
        auto extract_value = [](Iterator& input) -> std::string {
            std::string val{};
            while (*input != '{') {
                ++input;
            }
            ++input;
            while (*input != '}') {
                val.append(1, *input);
                ++input;
            }
            return val;
        };

        auto starts_with = [](Iterator& input, const char* value) -> bool {
            bool found{true};
            auto it = input;
            while (*value != '\0') {
                if (*it == *value) {
                    ++it;
                    ++value;
                } else {
                    found = false;
                    break;
                }
            }
            return found;
        };

        for (; it != end and *it != '}'; ++it) {

            if (ctx and starts_with(it, "{}")) {
                arg_id = ctx->next_arg_id();
                std::advance(it, 2);
                return it;
            }

            if (not std::isalpha(*it)) {
                continue;
            }

            if (starts_with(it, "p")) {
                auto val = extract_value(it);
                if (val == "f") {
                    format_.precision = Eigen::FullPrecision;
                } else if (val == "s") {
                    format_.precision = Eigen::StreamPrecision;
                } else {
                    format_.precision = std::stoi(val.c_str());
                }
            } else if (starts_with(it, "csep")) {
                format_.coeff_sep = extract_value(it);
            } else if (starts_with(it, "rsep")) {
                format_.row_sep = extract_value(it);
            } else if (starts_with(it, "rpre")) {
                format_.row_prefix = extract_value(it);
            } else if (starts_with(it, "rsuf")) {
                format_.row_suffix = extract_value(it);
            } else if (starts_with(it, "mpre")) {
                format_.mat_prefix = extract_value(it);
            } else if (starts_with(it, "msuf")) {
                format_.mat_suffix = extract_value(it);
            } else if (starts_with(it, "t")) {
                format_.transpose = true;
            } else if (starts_with(it, "noal")) {
                format_.dont_align_cols = true;
                std::advance(it, 3);
            } else {
                std::string token;
                while (std::isalpha(*it) and it != end) {
                    token += *it;
                    ++it;
                }
                throw format_error(
                    "invalid format, only p{int/str}, cesp{str}, rsep{str}, "
                    "rpre{str}, rsuf{str}, mpre{str}, msuf{str}, t and noal "
                    "are allowed. Found: " +
                    token);
            }
        }
        return it;
    }

    //! \brief Formats the given data to the context output iterator
    //!
    //! \param data The data to format
    //! \param ctx A formatting context
    //! \return decltype(ctx.out()) The iterator pointer after the last
    //! character parsed
    auto format(const T& data, format_context& ctx) -> decltype(ctx.out()) {
        if (arg_id == -1) {
#ifdef EIGEN_FMT_USE_IOFORMAT
            auto formatter = Eigen::IOFormat{
                format_.precision,
                format_.dont_align_cols ? Eigen::DontAlignCols : 0,
                format_.coeff_sep,
                format_.row_sep,
                format_.row_prefix,
                format_.row_suffix,
                format_.mat_prefix,
                format_.mat_suffix};
            std::stringstream ss;
            if (format_.transpose) {
                ss << data.transpose().format(formatter);
                // return print_matrix(data.transpose(), ctx);
            } else {
                // return print_matrix(data, ctx);
                ss << data.format(formatter);
            }
            return format_to(ctx.out(), "{}", ss.str());
#else
            return print_matrix(data, ctx);
#endif
        } else {
            visitor.ctx = &ctx;
            visitor.self = this;
            visitor.data = std::addressof(data);
            return visit_format_arg(visitor, ctx.arg(arg_id));
        }
    }

    //! \brief Native fmt implementation of the Eigen::print_matrix function
    //!
    //! \tparam U The type of the matrix to print
    //! \param data The matrix to print
    //! \param ctx A formatting context
    //! \return decltype(ctx.out()) The iterator pointer after the last
    //! character parsed
    template <typename U>
    auto print_matrix(const U& data, format_context& ctx)
        -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{}", EigenFmt::format(data, format_));
    }

    EigenFmt::FormatSpec format_;
    int arg_id{-1};

    //! \brief A visitor class handling dynamic format specification
    //!
    //! Example: fmt::print("{:{}}", mat, format_str);
    struct Visitor {
        //! \brief Operator called if the parameter is a const char*
        //!
        template <typename U>
        auto operator()(U arg) -> typename std::enable_if<
            std::is_same<U, char const*>::value,
            decltype(std::declval<format_context>().out())>::type {
            self->internal_parse(arg, arg + std::strlen(arg));
            self->arg_id = -1;
            return self->format(*data, *ctx);
        }

        //! \brief Operator called if the parameter is a string view
        //!
        template <typename U>
        auto operator()(U arg) -> typename std::enable_if<
            std::is_same<U, fmt::basic_string_view<char>>::value,
            decltype(std::declval<format_context>().out())>::type {
            return this->operator()(arg.data());
        }

        //! \brief Operator called if the parameter is an unsupported type
        //!
        template <typename U>
        auto operator()(U arg) -> typename std::enable_if<
            not std::is_same<U, char const*>::value and
                not std::is_same<U, fmt::basic_string_view<char>>::value,
            decltype(std::declval<format_context>().out())>::type {
            (void)arg;
            throw format_error(
                "Expected a string-like argument as format specifier");
            return ctx->out();
        }

        format_context* ctx{nullptr};
        formatter<T, char>* self{nullptr};
        const T* data{nullptr};
    };
    Visitor visitor;
};

} // namespace fmt
