#include "CLI/CLI.hpp"
#include <LBFGS.h>
#include <LBFGSpp/Param.h>
#include <trajan/core/trajectory.h>
#include <trajan/main/trajan_opt.h>

#ifdef OCC_HAVE_TBLITE
#include <occ/xtb/tblite_wrapper.h>

namespace trajan::main {

static const std::map<std::string, XTBModel::Type> type_map{
    {"gfn0xtb", XTBModel::Type::GFN0xTB},
    {"gfn1xtb", XTBModel::Type::GFN1xTB},
    {"gfn2xtb", XTBModel::Type::GFN2xTB},
    {"gfnff", XTBModel::Type::GFNFF}};

static std::map<XTBModel::Type, std::string> reverse_type_map() {
  std::map<XTBModel::Type, std::string> reverse_type_map;
  for (const auto &[key, value] : type_map) {
    reverse_type_map[value] = key;
  }
  return reverse_type_map;
};
static const auto reversed_type_map = reverse_type_map();

void run_opt_subcommand(const OPTOpts &opts) {

  core::Trajectory trajectory;

  // TODO: add flag to call load_files_into_memory() if desired
  trajectory.load_files(opts.infiles);

  LBFGSpp::LBFGSParam<double> param;
  param.epsilon = 1e-4;
  param.epsilon_rel = 1e-4;
  param.max_iterations = 100;
  LBFGSpp::LBFGSSolver<double> solver(param);

  class Objective {
  private:
    int niter = 0;
    core::Frame &m_frame;
    energy::XTBModel &m_em;

  public:
    Objective(core::Frame &frame, energy::XTBModel &em)
        : m_frame(frame), m_em(em) {}
    double operator()(const occ::Vec &x, occ::Vec &grad) {
      for (int i = 0; i < m_frame.num_atoms(); i++) {
        occ::Vec3 pos{x[3 * i], x[3 * i + 1], x[3 * i + 2]};
        m_frame.update_atom_position(i, pos);
      }
      auto sp = m_em.single_point(m_frame);
      grad = Eigen::Map<occ::Vec>(sp.grads.data(), sp.grads.size());
      trajan::log::info("  iteration {:3}: {:16.6f} kJ/mol", niter, sp.energy);
      niter++;
      return sp.energy;
    }
  };

  std::string model = reversed_type_map.at(opts.energy_model_type);

  while (trajectory.next_frame()) {
    core::Frame &frame = trajectory.frame();

    trajan::log::info("Initialising frame {} with energy model {}",
                      trajectory.current_frame_index(), model);
    XTBModel em;
    auto sp = em.single_point(frame);

    trajan::log::info("Starting energy {:8.6f} kJ/mol", sp.energy);
    trajan::log::info("Beginning minimisation with LBFGS");
    trajan::log::info(" threshold = {}", opts.epsilon);
    trajan::log::info(" max iterations = {}\n", opts.max_iter);
    em.set_verbosity(XTB_VERBOSITY_MUTED);

    Objective obj(frame, em);
    double fx = 0.0;
    occ::Vec x = frame.cart_pos_flat();
    int niter = solver.minimize(obj, x, fx);
    trajan::log::info("\nMinimised structure in {} iterations", niter);
    trajan::log::info("Final XTB summary: ");
    for (int i = 0; i < frame.num_atoms(); i++) {
      occ::Vec3 pos{x[3 * i], x[3 * i + 1], x[3 * i + 2]};
      frame.update_atom_position(i, pos);
    }
    em.set_verbosity(XTB_VERBOSITY_FULL);
    sp = em.single_point(frame);
    trajan::log::info("Final energy {:8.6f} kJ/mol", sp.energy);
    std::string of =
        fmt::format("optimised_{}.pdb", trajectory.current_frame_index());
    trajectory.set_output_file(of);
    trajectory.write_frame();
  }
}

CLI::App *add_opt_subcommand(CLI::App &app) {
  CLI::App *opt = app.add_subcommand("opt", "Optimise stucture/trajectory");
  auto opts = std::make_shared<OPTOpts>();
  opt->add_option("-t,--tr,--traj", opts->infiles, "Input trajectory file name")
      ->required()
      ->check(CLI::ExistingFile);
  opt->add_option("--o,--out", opts->outfile, "Output file for opt data")
      ->capture_default_str();
  opt->add_option("-m,--model", opts->energy_model_type, "Set the energy model")
      ->transform(CLI::CheckedTransformer(type_map, CLI::ignore_case));
  opt->add_option("--epsilon", opts->epsilon, "Energy cutoff")
      ->capture_default_str();
  opt->add_option("--maxiter", opts->max_iter, "Maximum number of iterations")
      ->capture_default_str();

  // TODO: In the future optimise just a selection
  //  std::string sel1 = "--s1,--sel1";
  //  opt->add_option(sel1, opts->raw_sel1,
  //                  "First selection (prefix: i=atom indices, a=atom types,
  //                  " "j=molecule indices, m=molecule types)\n"
  //                  "Examples:\n"
  //                  "  i1,2,3-5    (atom indices 1,2,3,4,5)\n"
  //                  "  aC,N,O      (atom types C, N, O)\n"
  //                  "  j1,3-5      (molecule indices 1,3,4,5)\n"
  //                  "  mM1,M2      (molecule types M1,M2)")
  //      ->required()
  //      ->check(io::selection_validator(opts->parsed_sel1));

  opt->callback([opts]() { run_opt_subcommand(*opts); });
  return opt;
}

} // namespace trajan::main
