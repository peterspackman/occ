/* 
 *      Copyright (c) 2020 Robert Shaw
 *		This file is a part of Libecpint.
 *
 *      Permission is hereby granted, free of charge, to any person obtaining
 *      a copy of this software and associated documentation files (the
 *      "Software"), to deal in the Software without restriction, including
 *      without limitation the rights to use, copy, modify, merge, publish,
 *      distribute, sublicense, and/or sell copies of the Software, and to
 *      permit persons to whom the Software is furnished to do so, subject to
 *      the following conditions:
 *
 *      The above copyright notice and this permission notice shall be
 *      included in all copies or substantial portions of the Software.
 *
 *      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *      EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *      MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *      NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 *      LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 *      OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 *      WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef MULTIARR_HEAD
#define MULTIARR_HEAD

/**
  * \file multiarr.hpp
  * \brief Helpful lightweight multi-index arrays and tuples to make the code easier to write and test. 
  * 
  * TODO: It is possible these are slowing things down a bit, need to run benchmarks. 
  */

#include <vector>
#include <tuple> 
#include <algorithm>
#include <sstream>

namespace libecpint {

	using Pair = std::tuple<int, int>;
	using Triple = std::tuple<int, int, int>;
	using Quintuple = std::tuple<int, int, int, int, int>; 
	using Heptuple = std::tuple<int, int, int, int, int, int, int>; 
	
	namespace aux{
		template<std::size_t...> struct seq{};
		template<std::size_t N, std::size_t... Is>
		struct gen_seq : gen_seq<N-1, N-1, Is...>{};
		template<std::size_t... Is>
		struct gen_seq<0, Is...> : seq<Is...>{};
		template<class Ch, class Tr, class Tuple, std::size_t... Is>
		void print_tuple(std::basic_ostream<Ch,Tr>& os, Tuple const& t, seq<Is...>){
			using swallow = int[];
			(void)swallow{0, (void(os << (Is == 0? "" : ", ") << std::get<Is>(t)), 0)...};
		}
	} 
	
	template<class Ch, class Tr, class... Args>
	auto operator<<(std::basic_ostream<Ch, Tr>& os, std::tuple<Args...> const& t)
		-> std::basic_ostream<Ch, Tr>&
	{
		os << "(";
		aux::print_tuple(os, t, aux::gen_seq<sizeof...(Args)>());
		return os << ")";
	}
	
	/// Templated skeleton two index array for convenience
	template<typename T>
	struct TwoIndex {
		int dims[2];
		std::vector<T> data;
		T& operator()(const int i, const int j) { return data[i * dims[1] + j]; }
		T operator()(const int i, const int j) const { return data[i * dims[1] + j]; }
		void assign(int dim1, int dim2, T value) {
			dims[0] = dim1; dims[1] = dim2;
			data.resize(dim1 * dim2);
			std::fill(data.begin(), data.end(), value);
		}
		TwoIndex<T> transpose() const {
			TwoIndex<T> result(dims[1], dims[0]);
			for (int i = 0; i < dims[0]; i++) {
				for (int j = 0; j < dims[1]; j++)
					result.data[j * dims[0] + i] = data[i * dims[1] + j];
			}
			return result;
		}
		void add(const TwoIndex<T>& other) {
			std::transform (data.begin(), data.end(), other.data.begin(), data.begin(), std::plus<T>());
		}
		
		void multiply(T k) {
			std::transform(data.begin(), data.end(), data.begin(), [&k](T& c){return c*k;});
		}
		TwoIndex() { dims[0] = dims[1] = 0; }
		TwoIndex(const int dim1, const int dim2) {
			dims[0] = dim1; dims[1] = dim2;
			data.resize(dim1 * dim2);
		}
		TwoIndex(const int dim1, const int dim2, const T value) { assign(dim1, dim2, value); }
		TwoIndex(const TwoIndex<T> &other) { 
			data = other.data;
			dims[0] = other.dims[0]; dims[1] = other.dims[1];
		}
	};

	/// Templated skeleton three index array for convenience
	template<typename T>
	struct ThreeIndex {
		int dims[3];
		std::vector<T> data;
		T& operator()(const int i, const int j, const int k) { return data[i*dims[2]*dims[1] + j*dims[2] + k]; }
		T operator()(const int i, const int j, const int k) const { return data[i*dims[2]*dims[1] + j*dims[2] + k]; }
		ThreeIndex(){ dims[0] = 0; dims[1] = 0; dims[2] = 0; }
		ThreeIndex(const int dim1, const int dim2, const int dim3) {
			dims[0] = dim1; dims[1] = dim2; dims[2] = dim3;
			data.resize(dim1 * dim2 * dim3);
		}
		ThreeIndex(const ThreeIndex<T> &other)  { 
			data = other.data;
			for (int n = 0; n < 3; n++) dims[n] = other.dims[n]; 
		}
		void fill(const T value) { std::fill(data.begin(), data.end(), value); }
	};

	/// Templated skeleton five index array for convenience
	template<typename T>
	struct FiveIndex {
		int dims[5];
		std::vector<T> data;
		T& operator()(const int i, const int j, const int k, const int l, const int m) {
			return data[m + dims[4] * (l + dims[3] * (k + dims[2] * (j + dims[1] * i)))]; 
		}
		T operator()(const int i, const int j, const int k, const int l, const int m) const {
			return data[m + dims[4] * (l + dims[3] * (k + dims[2] * (j + dims[1] * i)))];
		}
		FiveIndex() { dims[0] = dims[1] = dims[2] = dims[3] = dims[4] = 0; }
		FiveIndex(const int dim1, const int dim2, const int dim3, const int dim4, const int dim5) {
			dims[0] = dim1; dims[1] = dim2; dims[2] = dim3; dims[3] = dim4; dims[4] = dim5;
			data.resize(dim1 * dim2 * dim3 * dim4 * dim5);
		}
		FiveIndex(const FiveIndex<T> &other) { 
			data = other.data;
			for (int n = 0; n < 5; n++) dims[n] = other.dims[n]; 
		}
	};

	/// Templated skeleton seven index array for convenience
	template<typename T>
	struct SevenIndex {
		int dims[7];
		int mults[6];
		std::vector<T> data;
		T& operator()(const int i, const int j, const int k, const int l, const int m, const int n, const int p) {
			return data[p + mults[5]*n + mults[4]*m + mults[3]*l + mults[2]*k + mults[1]*j + mults[0]*i];
		}
		T operator()(const int i, const int j, const int k, const int l, const int m, const int n, const int p) const {
			return data[p + mults[5]*n + mults[4]*m + mults[3]*l + mults[2]*k + mults[1]*j + mults[0]*i];
		}
		SevenIndex() { dims[0] = dims[1] = dims[2] = dims[3] = dims[4] = dims[5] = dims[6] = 0; 
					   mults[0] = mults[1] = mults[2] = mults[3] = mults[4] = mults[5] = 0; }
		SevenIndex(const int dim1, const int dim2, const int dim3, const int dim4, const int dim5, const int dim6, const int dim7) {
			dims[0] = dim1; dims[1] = dim2; dims[2] = dim3; dims[3] = dim4; dims[4] = dim5; dims[5] = dim6; dims[6] = dim7;
			data.resize(dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * dim7);
			mults[5] = dim7;
			mults[4] = dim7*dim6;
			mults[3] = mults[4]*dim5;
			mults[2] = mults[3]*dim4;
			mults[1] = mults[2]*dim3;
			mults[0] = mults[1]*dim2;
		}
		SevenIndex(const SevenIndex<T> &other) {
			data = other.data;
			for (int n = 0; n < 6; n++) {
				dims[n] = other.dims[n];
				mults[n] = other.mults[n];
			}
			dims[6] = other.dims[6];
		}
	};

}
#endif
