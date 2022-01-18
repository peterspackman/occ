#include <occ/dft/seminumerical_exchange.h>
#include <occ/qm/ints.h>
#include <occ/core/parallel.h>
#include <occ/gto/gto.h>

namespace occ::dft::cosx {

using libint2::Operator;

SemiNumericalExchange::SemiNumericalExchange(const std::vector<occ::core::Atom> &atoms,
					     const qm::BasisSet &basis) : 
	m_atoms(atoms), m_basis(basis), m_grid(m_basis, m_atoms) {

    for (size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
    }
    std::tie(m_shellpair_list, m_shellpair_data) = occ::ints::compute_shellpairs(m_basis);
    m_overlap = occ::ints::compute_1body_ints<Operator::overlap>(m_basis, m_shellpair_list)[0];
    m_numerical_overlap = compute_overlap_matrix();
    occ::log::debug("Max error |Sn - S|: {:12.8f}\n", (m_numerical_overlap - m_overlap).array().cwiseAbs().maxCoeff());
    m_overlap_projector = m_numerical_overlap.ldlt().solve(m_overlap);
}

Mat SemiNumericalExchange::compute_overlap_matrix() const {
    size_t nbf = m_basis.nbf();
    using occ::parallel::nthreads;
    constexpr size_t BLOCKSIZE = 64;

    std::vector<Mat> SS(nthreads);
    for(size_t i = 0; i < nthreads; i++) {
	SS[i] = Mat::Zero(nbf, nbf);
    }
    for (const auto &atom_grid : m_atom_grids) {
	    const auto &atom_pts = atom_grid.points;
	    const auto &atom_weights = atom_grid.weights;
	    const size_t npt_total = atom_pts.cols();
	    const size_t num_blocks = npt_total / BLOCKSIZE + 1;

	    auto lambda = [&](int thread_id) {
		auto &S = SS[thread_id];
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
		    occ::gto::evaluate_basis(m_basis, m_atoms, pts_block,
					     ao, 0);
		    S.noalias() += ao.phi.transpose() * (ao.phi.array().colwise() * weights_block.array()).matrix();
		}
	    };
	    occ::parallel::parallel_do(lambda);
    }

    for(size_t i = 1; i < nthreads; i++) {
	SS[0].noalias() += SS[i];
    }
    return SS[0];
}

Mat SemiNumericalExchange::compute_K(qm::SpinorbitalKind kind, const qm::MolecularOrbitals &mo, double precision,
                     const Mat &Schwarz) const {
    size_t nbf = m_basis.nbf();
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
        libint2::Engine(Operator::nuclear, m_basis.max_nprim(), m_basis.max_l(), 0);


    Mat D2q = m_overlap_projector * D2;
    std::vector<Mat> KK(nthreads);
    for(size_t i = 0; i < nthreads; i++) {
	KK[i] = Mat::Zero(nbf, nbf);
    }
    const auto nshells = m_basis.size();
    const auto shell2bf = m_basis.shell2bf();

    // compute J, K
    for (const auto &atom_grid : m_atom_grids) {
            const auto &atom_pts = atom_grid.points;
            const auto &atom_weights = atom_grid.weights;
            const size_t npt_total = atom_pts.cols();
            const size_t num_blocks = npt_total / BLOCKSIZE + 1;

            auto lambda = [&](int thread_id) {
		std::vector<std::pair<double, std::array<double, 3>>> chgs(1);
                Mat rho(BLOCKSIZE, 1);
		auto &K = KK[thread_id];
		occ::gto::GTOValues ao;
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
		    occ::gto::evaluate_basis<0>(m_basis, m_atoms, pts_block, ao);
		    if(ao.phi.maxCoeff() < precision) continue;
		    Mat wao = ao.phi.array().colwise() * weights_block.array();
		    Mat Fg = wao * D2q;

		    Mat A(nbf, nbf);
		    Mat Gg(Fg.rows(), Fg.cols());
		    for(size_t pt = 0; pt < npt; pt++) {
			A.setZero();
			chgs[0] = {1, {pts_block(0, pt), pts_block(1, pt), pts_block(2, pt)}};
			engine.set_params(chgs);
			const auto &buf = engine.results();
			for (size_t s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
			    size_t bf1 = shell2bf[s1];
			    size_t n1 = m_basis[s1].size();
			    size_t s1_offset = s1 * (s1 + 1) / 2;
			    for (size_t s2 : m_shellpair_list.at(s1)) {
				size_t s12 = s1_offset + s2;
				size_t bf2 = shell2bf[s2];
				size_t n2 = m_basis[s2].size();
				engine.compute1(m_basis[s1], m_basis[s2]);
				Eigen::Map<const MatRM> buf_mat(buf[0], n1, n2);
				A.block(bf1, bf2, n1, n2) = buf_mat;
				if (s1 != s2) {
				    A.block(bf2, bf1, n2, n1) = 
					buf_mat.transpose();
				}
			    }
			}
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
    return 0.25 * (KK[0] + KK[0].transpose());
}

}
