#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/3rdparty/robin_hood.h>
#include <occ/core/atom.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>

namespace occ::dft {

using occ::qm::BasisSet;
using occ::qm::SpinorbitalKind;

using dfid = DensityFunctional::Identifier;

struct FuncComponent {
    dfid id;
    double factor{1.0};
    double hfx{0.0};
};

const robin_hood::unordered_map<std::string, std::vector<FuncComponent>>
    builtin_functionals({
        {"b3lyp", {{dfid::hyb_gga_xc_b3lyp, 1.0}}},
        {"b3pw91", {{dfid::hyb_gga_xc_b3pw91, 1.0}}},
        {"b3p86", {{dfid::hyb_gga_xc_b3p86, 1.0}}},
        {"o3lyp", {{dfid::hyb_gga_xc_o3lyp, 1.0}}},
        {"pbeh", {{dfid::hyb_gga_xc_pbeh}}},
        {"b97", {{dfid::hyb_gga_xc_b97}}},
        {"b971", {{dfid::hyb_gga_xc_b97_1}}},
        {"b972", {{dfid::hyb_gga_xc_b97_2}}},
        {"x3lyp", {{dfid::hyb_gga_xc_x3lyp}}},
        {"b97k", {{dfid::hyb_gga_xc_b97_k}}},
        {"b973", {{dfid::hyb_gga_xc_b97_3}}},
        {"mpw3pw", {{dfid::hyb_gga_xc_mpw3pw}}},
        {"mpw3lyp", {{dfid::hyb_gga_xc_mpw3lyp}}},
        {"bhandh", {{dfid::hyb_gga_xc_bhandh}}},
        {"bhandhlyp", {{dfid::hyb_gga_xc_bhandhlyp}}},
        {"b3lyp5", {{dfid::hyb_gga_xc_b3lyp5}}},
        {"pbe1pbe", {{dfid::gga_x_pbe, 0.75, 0.25}, {dfid::gga_c_pbe}}},
        {"pbe", {{dfid::gga_x_pbe, 1.0}, {dfid::gga_c_pbe, 1.0}}},
        {"pbepbe", {{dfid::gga_x_pbe, 1.0}, {dfid::gga_c_pbe, 1.0}}},
        {"pbe0", {{dfid::gga_x_pbe, 0.75, 0.25}, {dfid::gga_c_pbe}}},
        {"svwn", {{dfid::lda_x}, {dfid::lda_c_vwn_3}}},
        {"lda", {{dfid::lda_x}, {dfid::lda_c_vwn_3}}},
        {"lsda", {{dfid::lda_x}, {dfid::lda_c_vwn_3}}},
        {"svwn5", {{dfid::lda_x}, {dfid::lda_c_vwn}}},
        {"blyp", {{dfid::gga_x_b88}, {dfid::gga_c_lyp}}},
        {"bpbe", {{dfid::gga_x_b88}, {dfid::gga_c_pbe}}},
        {"bp86", {{dfid::gga_x_b88}, {dfid::gga_c_p86}}},
        {"m062x", {{dfid::hyb_mgga_x_m06_2x}, {dfid::mgga_c_m06_2x}}},
        {"tpss", {{dfid::mgga_x_tpss}, {dfid::mgga_c_tpss}}},
    });

int DFT::density_derivative() const {
    int deriv = 0;
    for (const auto &func : m_funcs) {
        deriv = std::max(deriv, func.derivative_order());
    }
    return deriv;
}

DFT::DFT(const std::string &method, const BasisSet &basis,
         const std::vector<occ::core::Atom> &atoms, SpinorbitalKind kind)
    : m_spinorbital_kind(kind), m_hf(atoms, basis), m_grid(basis, atoms) {
    occ::log::debug("start calculating atom grids... ");
    m_grid.set_max_angular_points(590);
    m_grid.set_min_angular_points(86);
    m_grid.set_radial_precision(1e-12);
    for (size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
    }
    size_t num_grid_points = std::accumulate(
        m_atom_grids.begin(), m_atom_grids.end(), 0.0,
        [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
    occ::log::debug("finished calculating atom grids ({} points)",
                    num_grid_points);
    occ::log::debug("Grid initialization took {} seconds",
                    occ::timing::total(occ::timing::grid_init));
    occ::log::debug("Grid point creation took {} seconds",
                    occ::timing::total(occ::timing::grid_points));
    m_funcs = parse_method(method,
                           m_spinorbital_kind == SpinorbitalKind::Unrestricted);
    for (const auto &func : m_funcs) {
        occ::log::debug(
            "Functional: {} {} {}, exact exchange = {}, polarized = {}",
            func.name(), func.kind_string(), func.family_string(),
            func.exact_exchange_factor(), func.polarized());
    }
    double hfx = exact_exchange_factor();
    if (hfx > 0.0)
        fmt::print("    {} x HF exchange\n", hfx);
}

std::vector<DensityFunctional> parse_method(const std::string &method_string,
                                            bool polarized) {
    std::vector<DensityFunctional> funcs;
    std::string method = occ::util::trim_copy(method_string);
    occ::util::to_lower(method);

    auto tokens = occ::util::tokenize(method_string, " ");
    fmt::print("Functionals:\n");
    for (const auto &token : tokens) {
        std::string m = token;
        occ::log::debug("Token: {}", m);
        if (m[0] == 'u')
            m = m.substr(1);
        if (builtin_functionals.contains(m)) {
            auto combo = builtin_functionals.at(m);
            occ::log::debug("Found builtin functional combination for {}", m);
            for (const auto &func : combo) {
                occ::log::debug("id: {}", func.id);
                auto f = DensityFunctional(func.id, polarized);
                occ::log::debug("scale factor: {}", func.factor);
                f.set_scale_factor(func.factor);
                fmt::print("    ");
                if (func.factor != 1.0)
                    fmt::print("{} x ", func.factor);
                fmt::print("{}\n", f.name());
                if (func.hfx > 0.0)
                    f.set_exchange_factor(func.hfx);
                funcs.push_back(f);
            }
        } else {
            fmt::print("    {}\n", token);
            funcs.push_back(DensityFunctional(token, polarized));
        }
    }
    return funcs;
}



Mat compute_a_matrix(const Mat &D, const qm::BasisSet &basis, const ints::shellpair_list_t &shellpair_list, const Vec3 &position, libint2::Engine &engine) {
    occ::timing::start(occ::timing::category::ints1e);

    using occ::qm::expectation;
    const auto n = basis.nbf();
    const auto nshells = basis.size();
    auto shell2bf = basis.shell2bf();
    Mat result = Mat::Zero(n, n);

    std::vector<std::pair<double, std::array<double, 3>>> chgs{{1, {position(0), position(1), position(2)}}};
    auto compute = [&](int thread_id) {
	engine.set_params(chgs);
        const auto &buf = engine.results();
        for (size_t s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
	    size_t bf1 = shell2bf[s1];
            size_t n1 = basis[s1].size();
            size_t s1_offset = s1 * (s1 + 1) / 2;
            for (size_t s2 : shellpair_list.at(s1)) {
		size_t s12 = s1_offset + s2;
                size_t bf2 = shell2bf[s2];
                size_t n2 = basis[s2].size();
                engine.compute(basis[s1], basis[s2]);

                Eigen::Map<const MatRM> buf_mat(buf[0], n1, n2);
                result.block(bf1, bf2, n1, n2) = buf_mat;
                if (s1 != s2) {
		    result.block(bf2, bf1, n2, n1) = 
			buf_mat.transpose();
                }
            }
        }
    };
    occ::parallel::parallel_do(compute);
    return result;
}

Mat DFT::compute_sgx_jk(SpinorbitalKind kind, const MolecularOrbitals &mo, double precision,
                     const Mat &Schwarz) const {
    const auto &basis = m_hf.basis();
    const auto &atoms = m_hf.atoms();
    size_t nbf = basis.nbf();
    const auto& D = mo.D;
    Mat D2 = 2 * D;
    constexpr size_t BLOCKSIZE = 64;
    using occ::parallel::nthreads;

    Mat Sn = Mat::Zero(nbf, nbf);

    typedef std::array<Mat, libint2::operator_traits<Operator::nuclear>::nopers>
        result_type;
    const unsigned int nopers =
        libint2::operator_traits<Operator::nuclear>::nopers;

    libint2::Engine engine =
        libint2::Engine(Operator::nuclear, basis.max_nprim(), basis.max_l(), 0);


    // Compute overlap metric
    size_t atom_grid_idx{0};
    for (const auto &atom_grid : m_atom_grids) {
            const auto &atom_pts = atom_grid.points;
            const auto &atom_weights = atom_grid.weights;
            const size_t npt_total = atom_pts.cols();
            const size_t num_blocks = npt_total / BLOCKSIZE + 1;

            auto lambda = [&](int thread_id) {
                Mat rho(BLOCKSIZE, 1);
                for (size_t block = 0; block < num_blocks; block++) {
                    if (block % nthreads != thread_id)
                        continue;
                    Eigen::Index l = block * BLOCKSIZE;
                    Eigen::Index u =
                        std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
                    Eigen::Index npt = u - l;
                    if (npt <= 0)
                        continue;

                    const auto &pts_block = atom_pts.middleCols(l, npt);
                    const auto &weights_block = atom_weights.segment(l, npt);
		    occ::gto::GTOValues ao;
		    occ::gto::evaluate_basis(basis, atoms, pts_block,
                                             ao, 0);
		    Sn.noalias() += ao.phi.transpose() * (ao.phi.array().colwise() * weights_block.array()).matrix();
		}
	    };
	    occ::parallel::parallel_do(lambda);
    }
    Mat S = m_hf.compute_overlap_matrix();
    fmt::print("Max error |Sn - S|: {:12.8f}\n", (Sn - S).array().cwiseAbs().maxCoeff());
    Mat Q = Sn.householderQr().solve(S);
    Mat D2q = Q * D2;
    std::vector<Mat> KK(nthreads);
    for(size_t i = 0; i < nthreads; i++) {
	KK[i] = Mat::Zero(nbf, nbf);
    }

    // compute J, K
    for (const auto &atom_grid : m_atom_grids) {
            const auto &atom_pts = atom_grid.points;
            const auto &atom_weights = atom_grid.weights;
            const size_t npt_total = atom_pts.cols();
            const size_t num_blocks = npt_total / BLOCKSIZE + 1;

            auto lambda = [&](int thread_id) {
                Mat rho(BLOCKSIZE, 1);
		auto &K = KK[thread_id];
                for (size_t block = 0; block < num_blocks; block++) {
                    if (block % nthreads != thread_id)
                        continue;
                    Eigen::Index l = block * BLOCKSIZE;
                    Eigen::Index u =
                        std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
                    Eigen::Index npt = u - l;
                    if (npt <= 0)
                        continue;

                    const auto &pts_block = atom_pts.middleCols(l, npt);
                    const auto &weights_block = atom_weights.segment(l, npt);
		    occ::gto::GTOValues ao;
		    occ::gto::evaluate_basis(basis, atoms, pts_block, ao, 0);
		    Mat wao = ao.phi.array().colwise() * weights_block.array();
		    Mat Fg = wao * D2q;

		    Mat A(nbf, nbf);
		    Mat Gg(Fg.rows(), Fg.cols());
		    for(size_t pt = 0; pt < npt; pt++) {
			A = compute_a_matrix(
			    D, basis, m_hf.shellpair_list(), pts_block.col(pt), engine
			);
			Gg.row(pt) = Fg.row(pt) * A;
		    }
		    K.noalias() -= ao.phi.transpose() * Gg;
		}
	    };
	    occ::parallel::parallel_do(lambda);
    }

    for(size_t i = 1; i < nthreads; i++) {
	KK[0] += KK[i];
    }

    //auto [Je, Ke] = m_hf.compute_JK(occ::qm::SpinorbitalKind::Restricted, mo, 1e-12, occ::Mat());
    //fmt::print("K_exact\n{}\n", Ke);

    Mat result = 0.25 * (KK[0] + KK[0].transpose());
    //fmt::print("max error |result - K| {}\n", (result.array() - Ke.array()).cwiseAbs().maxCoeff());
    return result;
}


} // namespace occ::dft
