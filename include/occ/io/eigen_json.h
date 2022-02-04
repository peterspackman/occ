#pragma once
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <fmt/core.h>

namespace Eigen {

template<typename ScalarType, int Rows, int Cols >
void to_json(nlohmann::json& j, const Matrix<ScalarType, Rows, Cols>& mat) {
    for(int r = 0; r < mat.rows(); r++) {
        nlohmann::json arr;
	const auto &row = mat.row(r);
        for(int c = 0; c < mat.cols(); ++c)
        {
            arr.push_back(row(c));
        }
	j.push_back(arr);
    }
}


template<typename ScalarType, int Rows, int Cols>
void from_json(const nlohmann::json& j, Matrix<ScalarType, Rows, Cols> &dest_mat) {
    nlohmann::json jcols;
    if(j.is_array()) {
	if(j.empty()) return;
	jcols = j;
    }
    else if(j.is_number()) {
	jcols.push_back(j);
    }
    else
    {
	throw "expected array or number for matrix conversion";
    }

    nlohmann::json jrows;
    if(jcols.front().is_array()) {
	jrows = jcols;
    }
    else {
	if constexpr(Rows == 1) {
	    jrows.push_back(jcols);
	}
	else if constexpr(Cols == 1) {
	    for (unsigned int i = 0; i < jcols.size(); ++i) {
		jrows.push_back({jcols.at(i)});
	    }
	}
	else {
	    throw "Expected a matrix, received a vector.";
	}
    }

    const auto rows = jrows.size();
    const auto cols = jrows.front().size();
    if((Rows >= 0 && static_cast<int>(rows) != Rows) || (Cols >= 0 && static_cast<int>(cols) != Cols)) {
	throw fmt::format(
	    "Expected matrix of size {} {}, received matrix of size {} {}",
	    Rows, Cols, rows, cols);
    }

    dest_mat.resize(rows, cols);
    for(int r = 0; r < rows; r++)
    {
	if(jrows.at(r).size() != cols) {
	    throw "inconsistent matrix size: some rows have different numbers of columns";
	}
	auto row = dest_mat.row(r);
	const auto &jrow = jrows.at(r);
	for(int c = 0; c < cols; c++) {
	    row(c) = jrow.at(c);
	}
    }
}
}
