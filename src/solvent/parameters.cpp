#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <occ/core/log.h>
#include <occ/io/solvent_json.h>
#include <occ/solvent/parameters.h>

namespace fs = std::filesystem;

namespace impl {
bool solvent_parameter_data_initialized{false};
static std::string data_path_directory_override{""};
} // namespace impl

namespace occ::solvent {

void override_data_path_directory(const std::string &s) {
  impl::data_path_directory_override = s;
}

std::string solvent_data_path() {
  std::string path{"."};
  const char *data_path_env = impl::data_path_directory_override.empty()
                                  ? getenv("OCC_DATA_PATH")
                                  : impl::data_path_directory_override.c_str();
  if (data_path_env) {
    path = data_path_env;
  }
  std::string solvent_data_path = path + std::string("/solvent");
  bool path_exists = fs::exists(solvent_data_path);
  std::string errmsg;
  if (!path_exists) { // try without "/solvent"
    occ::log::warn("There is a problem with the solvent data directory, the "
                   "path '{}' is not valid (does not exist)",
                   solvent_data_path);
    solvent_data_path = fs::current_path().string();
  } else if (!fs::is_directory(solvent_data_path)) {
    occ::log::warn("There is a problem with the solvent data directory, the "
                   "path '{}' is not valid (not a directory)",
                   solvent_data_path);
    solvent_data_path = fs::current_path().string();
  }
  return solvent_data_path;
}

using DielectricMap = ankerl::unordered_dense::map<std::string, double>;

static inline DielectricMap dielectric_constant{
    {"acetic acid", 6.2528},
    {"acetone", 20.493},
    {"acetonitrile", 35.688},
    {"acetophenone", 17.440},
    {"aniline", 6.8882},
    {"anisole", 4.2247},
    {"benzaldehyde", 18.220},
    {"benzene", 2.2706},
    {"benzonitrile", 25.592},
    {"benzyl chloride", 6.7175},
    {"1-bromo-2-methylpropane", 7.7792},
    {"bromobenzene", 5.3954},
    {"bromoethane", 9.01},
    {"bromoform", 4.2488},
    {"1-bromooctane", 5.0244},
    {"1-bromopentane", 6.269},
    {"2-bromopropane", 9.3610},
    {"1-bromopropane", 8.0496},
    {"butanal", 13.450},
    {"butanoic acid", 2.9931},
    {"1-butanol", 17.332},
    {"2-butanol", 15.944},
    {"butanone", 18.246},
    {"butanonitrile", 24.291},
    {"butyl acetate", 4.9941},
    {"butylamine", 4.6178},
    {"n-butylbenzene", 2.360},
    {"sec-butylbenzene", 2.3446},
    {"tert-butylbenzene", 2.3447},
    {"carbon disulfide", 2.6105},
    {"carbon tetrachloride", 2.2280},
    {"chlorobenzene", 5.6968},
    {"sec-butyl chloride", 8.3930},
    {"chloroform", 4.7113},
    {"1-chlorohexane", 5.9491},
    {"1-chloropentane", 6.5022},
    {"1-chloropropane", 8.3548},
    {"o-chlorotoluene", 4.6331},
    {"m-cresol", 12.440},
    {"o-cresol", 6.760},
    {"cyclohexane", 2.0165},
    {"cyclohexanone", 15.619},
    {"cyclopentane", 1.9608},
    {"cyclopentanol", 16.989},
    {"cyclopentanone", 13.58},
    {"cis-decalin", 2.2139},
    {"trans-decalin", 2.1781},
    {"decalin (cis-trans mixture)", 2.196},
    {"n-decane", 1.9846},
    {"1-decanol", 7.5305},
    {"1,2-dibromoethane", 4.9313},
    {"dibromomethane", 7.2273},
    {"dibutyl ether", 3.0473},
    {"o-dichlorobenzene", 9.9949},
    {"1,2-dichloroethane", 10.125},
    {"cis-dichloroethylene", 9.200},
    {"trans-dichloroethylene", 2.140},
    {"dichloromethane", 8.930},
    {"diethyl ether", 4.2400},
    {"diethyl sulfide", 5.723},
    {"diethylamine", 3.5766},
    {"diiodomethane", 5.320},
    {"diisopropyl ether", 3.380},
    {"dimethyl disulfide", 9.600},
    {"dimethylsulfoxide", 46.826},
    {"N,N-dimethylacetamide", 37.781},
    {"cis-1,2-dimethylcyclohexane", 2.060},
    {"N,N-dimethylformamide", 37.219},
    {"2,4-dimethylpentane", 1.8939},
    {"2,4-dimethylpyridine", 9.4176},
    {"2,6-dimethylpyridine", 7.1735},
    {"1,4-dioxane", 2.2099},
    {"diphenyl ether", 3.730},
    {"dipropylamine", 2.9112},
    {"n-dodecane", 2.0060},
    {"1,2-ethanediol", 40.245},
    {"ethanethiol", 6.667},
    {"ethanol", 24.852},
    {"ethyl acetate", 5.9867},
    {"ethyl formate", 8.3310},
    {"ethylbenzene", 2.4339},
    {"ethylphenyl ether", 4.1797},
    {"fluorobenzene", 5.420},
    {"1-fluorooctane", 3.890},
    {"formamide", 108.94},
    {"formic acid", 51.100},
    {"n-heptane", 1.9113},
    {"1-heptanol", 11.321},
    {"2-heptanone", 11.658},
    {"4-heptanone", 12.257},
    {"n-hexadecane", 2.0402},
    {"n-hexane", 1.8819},
    {"hexanoic acid", 2.600},
    {"1-hexanol", 12.51},
    {"2-hexanone", 14.136},
    {"1-hexene", 2.0717},
    {"1-hexyne", 2.615},
    {"iodobenzene", 4.5470},
    {"1-iodobutane", 6.173},
    {"iodoethane", 7.6177},
    {"1-iodohexadecane", 3.5338},
    {"iodomethane", 6.8650},
    {"1-iodopentane", 5.6973},
    {"1-iodopropane", 6.9626},
    {"isopropylbenzene", 2.3712},
    {"p-isopropyltoluene", 2.2322},
    {"mesitylene", 2.2650},
    {"methanol", 32.613},
    {"2-methoxyethanol", 17.200},
    {"methyl acetate", 6.8615},
    {"methyl benzoate", 6.7367},
    {"methyl butanoate", 5.5607},
    {"methyl formate", 8.8377},
    {"4-methyl-2-pentanone", 12.887},
    {"methyl propanoate", 6.0777},
    {"2-methyl-1-propanol", 16.777},
    {"2-methyl-2-propanol", 12.470},
    {"N-methylaniline", 5.9600},
    {"methylcyclohexane", 2.024},
    {"N-methylformamide (E-Z mixture)", 181.56},
    {"2-methylpentane", 1.890},
    {"2-methylpyridine", 9.9533},
    {"3-methylpyridine", 11.645},
    {"4-methylpyridine", 11.957},
    {"nitrobenzene", 34.809},
    {"nitroethane", 28.290},
    {"nitromethane", 36.562},
    {"1-nitropropane", 23.730},
    {"2-nitropropane", 25.654},
    {"o-nitrotoluene", 25.669},
    {"n-nonane", 1.9605},
    {"1-nonanol", 8.5991},
    {"5-nonanone", 10.600},
    {"n-octane", 1.9406},
    {"1-octanol", 9.8629},
    {"2-octanone", 9.4678},
    {"n-pentadecane", 2.0333},
    {"pentanal", 10.0},
    {"n-pentane", 1.8371},
    {"pentanoic acid", 2.6924},
    {"1-pentanol", 15.130},
    {"2-pentanone", 15.200},
    {"3-pentanone", 16.780},
    {"1-pentene", 1.9905},
    {"E-2-pentene", 2.051},
    {"pentyl acetate", 4.7297},
    {"pentylamine", 4.2010},
    {"perfluorobenzene", 2.029},
    {"phenylmethanol", 12.457},
    {"propanal", 18.500},
    {"propanoic acid", 3.440},
    {"1-propanol", 20.524},
    {"2-propanol", 19.264},
    {"propanonitrile", 29.324},
    {"2-propen-1-ol", 19.011},
    {"propyl acetate", 5.5205},
    {"propylamine", 4.9912},
    {"pyridine", 12.978},
    {"tetrachloroethene", 2.268},
    {"tetrahydrofuran", 7.4257},
    {"tetrahydrothiophene-S,S-dioxide", 43.962},
    {"tetralin", 2.771},
    {"thiophene", 2.7270},
    {"thiophenol", 4.2728},
    {"toluene", 2.3741},
    {"tributyl phosphate", 8.1781},
    {"1,1,1-trichloroethane", 7.0826},
    {"1,1,2-trichloroethane", 7.1937},
    {"trichloroethene", 3.422},
    {"triethylamine", 2.3832},
    {"2,2,2-trifluoroethanol", 26.726},
    {"1,2,4-trimethylbenzene", 2.3653},
    {"2,2,4-trimethylpentane", 1.9358},
    {"n-undecane", 1.9910},
    {"m-xylene", 2.3478},
    {"o-xylene", 2.5454},
    {"p-xylene", 2.2705},
    {"xylene", 2.3879},
    {"water", 78.400},
};

using SMDParameterMap =
    ankerl::unordered_dense::map<std::string, SMDSolventParameters>;
/*
 * Parameters taken from https://comp.chem.umn.edu/solvation/mnsddb.pdf
 */
static inline SMDParameterMap smd_solvent_parameters{
    {"1,1,1-trichloroethane",
     SMDSolventParameters(1.4379, 1.4313, 0.0, 0.09, 36.24, 7.0826, 0.0, 0.60)},
    {"1,1,2-trichloroethane", SMDSolventParameters(1.4717, 1.4689, 0.13, 0.13,
                                                   48.97, 7.1937, 0.0, 0.60)},
    {"1,2,4-trimethylbenzene", SMDSolventParameters(1.5048, 1.5024, 0.0, 0.19,
                                                    42.03, 2.3653, 0.667, 0.0)},
    {"1,2-dibromoethane",
     SMDSolventParameters(1.5387, 1.5364, 0.10, 0.17, 56.93, 4.9313, 0.0, 0.5)},
    {"1,2-dichloroethane",
     SMDSolventParameters(1.4448, 1.4425, 0.10, 0.11, 45.86, 10.125, 0.0, 0.5)},
    {"1,2-ethanediol",
     SMDSolventParameters(1.4318, 1.4306, 0.58, 0.78, 69.07, 40.245, 0.0, 0.5)},
    {"1,4-dioxane",
     SMDSolventParameters(1.4224, 1.4204, 0.00, 0.64, 47.14, 2.2099, 0.0, 0.0)},
    {"1-bromo-2-methylpropane",
     SMDSolventParameters(1.4348, 1.4349, 0.00, 0.12, 34.69, 7.7792, 0.0, 0.2)},
    {"1-bromooctane", SMDSolventParameters(1.4524, 1.4500, 0.0, 0.12, 41.28,
                                           5.0244, 0.0, 0.111)},
    {"1-bromopentane",
     SMDSolventParameters(1.4447, 1.4420, 0.00, 0.12, 38.7, 6.269, 0.0, 0.250)},
    {"1-bromopropane", SMDSolventParameters(1.4343, 1.4315, 0.0, 0.12, 36.36,
                                            8.0496, 0.0, 0.250)},
    {"1-butanol",
     SMDSolventParameters(1.3993, 1.3971, 0.37, 0.48, 35.88, 17.332, 0.0, 0.0)},
    {"1-chlorohexane",
     SMDSolventParameters(1.4199, -1, 0.0, 0.10, 37.03, 5.9491, 0.0, 0.143)},
    {"1-chloropentane",
     SMDSolventParameters(1.4127, 1.4104, 0.0, 0.1, 35.12, 6.5022, 0.0, 0.167)},
    {"1-chloropropane",
     SMDSolventParameters(1.3879, 1.3851, 0.0, 0.1, 30.66, 8.3548, 0.0, 0.25)},
    {"1-decanol",
     SMDSolventParameters(1.4372, 1.4353, 0.37, 0.48, 41.04, 7.5305, 0.0, 0.0)},
    {"1-fluorooctane",
     SMDSolventParameters(1.3935, 1.3927, 0.0, 0.10, 33.92, 3.89, 0.0, 0.111)},
    {"1-heptanol",
     SMDSolventParameters(1.4249, 1.4224, 0.37, 0.48, 38.5, 11.321, 0.0, 0.0)},
    {"1-hexanol",
     SMDSolventParameters(1.4178, 1.4162, 0.37, 0.48, 37.15, 12.51, 0.0, 0.0)},
    {"1-hexene",
     SMDSolventParameters(1.3837, 1.385, 0.00, 0.07, 25.76, 2.0717, 0.0, 0.0)},
    {"1-hexyne",
     SMDSolventParameters(1.3989, 1.3957, 0.12, 0.10, 28.79, 2.615, 0.0, 0.0)},
    {"1-iodobutane",
     SMDSolventParameters(1.5001, 1.4958, 0.00, 0.15, 40.65, 6.173, 0.0, 0.0)},
    {"1-iodohexadecane",
     SMDSolventParameters(1.4806, -1, 0.00, 0.15, 46.48, 3.5338, 0.0, 0.0)},
    {"1-iodopentane",
     SMDSolventParameters(1.4959, -1, 0.00, 0.15, 41.56, 5.6973, 0.0, 0.0)},
    {"1-iodopropane",
     SMDSolventParameters(1.5058, 1.5027, 0.00, 0.15, 41.45, 6.9626, 0.0, 0.0)},
    {"1-nitropropane",
     SMDSolventParameters(1.4018, 1.3996, 0.00, 0.31, 43.32, 23.73, 0.0, 0.0)},
    {"1-nonanol",
     SMDSolventParameters(1.4333, 1.4319, 0.37, 0.48, 40.14, 8.5991, 0.0, 0.0)},
    {"1-octanol",
     SMDSolventParameters(1.4295, 1.4279, 0.37, 0.48, 39.01, 9.8629, 0.0, 0.0)},
    {"1-pentanol",
     SMDSolventParameters(1.4101, 1.4080, 0.37, 0.48, 36.5, 15.13, 0.0, 0.0)},
    {"1-pentene",
     SMDSolventParameters(1.3715, 1.3684, 0.00, 0.07, 22.24, 1.9905, 0.0, 0.0)},
    {"1-propanol",
     SMDSolventParameters(1.3850, 1.3837, 0.37, 0.48, 33.57, 20.524, 0.0, 0.0)},
    {"2,2,2-trifluoroethanol",
     SMDSolventParameters(1.2907, -1, 0.57, 0.25, 42.02, 26.726, 0.0, 0.5)},
    {"2,2,4-trimethylpentane",
     SMDSolventParameters(1.3915, 1.3889, 0.00, 0.00, 26.38, 1.9358, 0.0, 0.0)},
    {"2,4-dimethylpentane",
     SMDSolventParameters(1.3815, 1.3788, 0.0, 0.00, 25.42, 1.8939, 0.0, 0.0)},
    {"2,4-dimethylpyridine", SMDSolventParameters(1.5010, 1.4985, 0.0, 0.63,
                                                  46.86, 9.4176, 0.625, 0.0)},
    {"2,6-dimethylpyridine", SMDSolventParameters(1.4953, 1.4952, 0.0, 0.63,
                                                  44.64, 7.1735, 0.625, 0.0)},
    {"2-bromopropane", SMDSolventParameters(1.4251, 1.4219, 0.00, 0.14, 33.46,
                                            9.3610, 0.0, 0.25)},
    {"2-butanol",
     SMDSolventParameters(1.3978, 1.3949, 0.33, 0.56, 32.44, 15.944, 0.0, 0.0)},
    {"2-chlorobutane",
     SMDSolventParameters(1.3971, 1.3941, 0.00, 0.12, 31.1, 8.3930, 0.0, 0.2)},
    {"2-heptanone",
     SMDSolventParameters(1.4088, 1.4073, 0.0, 0.51, 37.6, 11.658, 0.0, 0.0)},
    {"2-hexanone",
     SMDSolventParameters(1.4007, 1.3987, 0.0, 0.51, 36.63, 14.136, 0.0, 0.0)},
    {"2-methoxyethanol",
     SMDSolventParameters(1.4024, 1.4003, 0.30, 0.84, 44.39, 17.2, 0.0, 0.0)},
    {"2-methyl-1-propanol",
     SMDSolventParameters(1.3955, 1.3938, 0.37, 0.48, 32.38, 16.777, 0.0, 0.0)},
    {"2-methyl-2-propanol",
     SMDSolventParameters(1.3878, 1.3852, 0.31, 0.60, 28.73, 12.47, 0.0, 0.0)},
    {"2-methylpentane",
     SMDSolventParameters(1.3715, 1.3687, 0.0, 0.00, 24.3, 1.89, 0.0, 0.0)},
    {"2-methylpyridine",
     SMDSolventParameters(1.4957, 1.4984, 0.0, 0.58, 47.5, 9.9533, 0.714, 0.0)},
    {"2-nitropropane", SMDSolventParameters(1.3944, 1.3923, 0.00, 0.33, 42.16,
                                            25.654, 0.00, 0.0)},
    {"2-octanone",
     SMDSolventParameters(1.4151, 1.4133, 0.00, 0.51, 37.29, 9.4678, 0.0, 0.0)},
    {"2-pentanone",
     SMDSolventParameters(1.3895, 1.3885, 0.00, 0.51, 33.46, 15.200, 0.0, 0.0)},
    {"2-propanol",
     SMDSolventParameters(1.3776, 1.3752, 0.33, 0.56, 30.13, 19.264, 0.0, 0.0)},
    {"2-propen-1-ol",
     SMDSolventParameters(1.4135, 0.00, 0.38, 0.48, 36.39, 19.011, 0.0, 0.0)},
    {"E-2-pentene",
     SMDSolventParameters(1.3793, 1.3761, 0.00, 0.07, 23.62, 2.051, 0.0, 0.0)},
    {"3-methylpyridine", SMDSolventParameters(1.5040, 1.5043, 0.0, 0.54, 49.61,
                                              11.645, 0.714, 0.0)},
    {"3-pentanone",
     SMDSolventParameters(1.3924, 1.3905, 0.00, 0.51, 35.61, 16.78, 0.0, 0.0)},
    {"4-heptanone",
     SMDSolventParameters(1.4069, 1.4045, 0.00, 0.51, 35.98, 12.257, 0.0, 0.0)},
    {"4-methyl-2-pentanone",
     SMDSolventParameters(1.3962, 1.394, 0.0, 0.51, 33.83, 12.887, 0.0, 0.0)},
    {"4-methylpyridine",
     SMDSolventParameters(1.5037, 1.503, 0.0, 0.54, 50.17, 11.957, 0.714, 0.0)},
    {"5-nonanone",
     SMDSolventParameters(1.4195, 0.00, 0.0, 0.51, 37.83, 10.6, 0.0, 0.0)},
    {"acetic acid",
     SMDSolventParameters(1.3720, 1.3698, 0.61, 0.44, 39.01, 6.2528, 0.0, 0.0)},
    {"acetone",
     SMDSolventParameters(1.3588, 1.3559, 0.04, 0.49, 33.77, 20.493, 0.0, 0.0)},
    {"acetonitrile",
     SMDSolventParameters(1.3442, 1.3416, 0.07, 0.32, 41.25, 35.688, 0.0, 0.0)},
    {"acetophenone", SMDSolventParameters(1.5372, 1.5321, 0.00, 0.48, 56.19,
                                          17.44, 0.667, 0.0)},
    {"aniline", SMDSolventParameters(1.5863, 1.5834f, 0.26, 0.41, 60.62, 6.8882,
                                     0.857, 0.0)},
    {"anisole",
     SMDSolventParameters(1.5174, 1.5143, 0.0, 0.29, 50.52, 4.2247, 0.75, 0.0)},
    {"benzaldehyde", SMDSolventParameters(1.5463, 1.5433, 0.0, 0.39, 54.69,
                                          18.220, 0.857, 0.0)},
    {"benzene",
     SMDSolventParameters(1.5011, 1.4972, 0.0, 0.14, 40.62, 2.2706, 1.0, 0.0)},
    {"benzonitrile",
     SMDSolventParameters(1.5289, 1.5257, 0.0, 0.33, 55.83, 25.592, 0.75, 0.0)},
    {"benzylalcohol", SMDSolventParameters(1.5396, 1.5384, 0.33, 0.56, 52.96,
                                           12.457, 0.75, 0.0)},
    {"bromobenzene", SMDSolventParameters(1.5597, 1.5576, 0.0, 0.09, 50.72,
                                          5.3954, 0.857, 0.143)},
    {"bromoethane",
     SMDSolventParameters(1.4239, 1.4187f, 0.0, 0.12, 34.0, 9.01, 0.0, 0.333)},
    {"bromoform", SMDSolventParameters(1.6005, 1.5956, 0.15, 0.06, 64.58,
                                       4.2488, 0.0, 0.75)},
    {"butanal",
     SMDSolventParameters(1.3843, 1.3766, 0.0, 0.45, 35.06, 13.45, 0.0, 0.0)},
    {"butanoic acid",
     SMDSolventParameters(1.3980, 1.3958, 0.60, 0.45, 37.49, 2.9931, 0.0, 0.0)},
    {"butanone",
     SMDSolventParameters(1.3788, 1.3764, 0.00, 0.51, 34.5, 18.246, 0.0, 0.0)},
    {"butanonitrile",
     SMDSolventParameters(1.3842, 1.382, 0.0, 0.36, 38.75, 24.291, 0.0, 0.0)},
    {"butylethanoate",
     SMDSolventParameters(1.3941, 1.3923, 0.0, 0.45, 35.81, 4.9941, 0.0, 0.0)},
    {"butylamine",
     SMDSolventParameters(1.4031, 1.3987, 0.16, 0.61, 33.74, 4.6178, 0.0, 0.0)},
    {"n-butylbenzene",
     SMDSolventParameters(1.4898, 1.4874, 0.0, 0.15, 41.33, 2.36, 0.6, 0.0)},
    {"sec-butylbenzene",
     SMDSolventParameters(1.4895, 1.4878, 0.0, 0.16, 40.35, 2.3446, 0.60, 0.0)},
    {"tert-butylbenzene",
     SMDSolventParameters(1.4927, 1.4902, 0.0, 0.16, 39.78, 2.3447, 0.6, 0.0)},
    {"carbon disulfide",
     SMDSolventParameters(1.6319, 1.6241, 0.0, 0.07, 45.45, 2.6105, 0.0, 0.0)},
    {"carbon tetrachloride",
     SMDSolventParameters(1.4601, 1.4574, 0.00, 0.00, 38.04, 2.2280, 0.0, 0.8)},
    {"chlorobenzene", SMDSolventParameters(1.5241, 1.5221, 0.0, 0.07, 47.48,
                                           5.6968, 0.857, 0.143)},
    {"chloroform", SMDSolventParameters(1.4459, 1.4431, 0.15, 0.02, 38.39,
                                        4.7113, 0.0, 0.75)},
    {"a-chlorotoluene",
     SMDSolventParameters(1.5391, 0.0, 0.0, 0.33, 53.04, 6.7175, 0.75, 0.125)},
    {"o-chlorotoluene", SMDSolventParameters(1.5268, 1.5233, 0.00, 0.07, 47.43,
                                             4.6331, 0.75, 0.125)},
    {"m-cresol",
     SMDSolventParameters(1.5438, 1.5394, 0.57, 0.34, 51.37, 12.44, 0.75, 0.0)},
    {"o-cresol",
     SMDSolventParameters(1.5361, 1.5399, 0.52, 0.30, 53.11, 6.76, 0.75, 0.0)},
    {"cyclohexane",
     SMDSolventParameters(1.4266, 1.4235, 0.00, 0.00, 35.48, 2.0165, 0.0, 0.0)},
    {"cyclohexanone", SMDSolventParameters(1.4507, 1.4507, 0.00, 0.56, 49.76,
                                           15.619, 0.00, 0.0)},
    {"cyclopentane",
     SMDSolventParameters(1.4065, 1.4036, 0.00, 0.00, 31.49, 1.9608, 0.0, 0.0)},
    {"cyclopentanol",
     SMDSolventParameters(1.4530, -1, 0.32, 0.56, 46.8, 16.989, 0.0, 0.0)},
    {"cyclopentanone",
     SMDSolventParameters(1.4366, 1.4347, 0.00, 0.52, 47.21, 13.58, 0.0, 0.0)},
    {"decalin (cis-trans mixture)",
     SMDSolventParameters(1.4753, 1.472, 0.00, 0.00, 43.82, 2.196, 0.0, 0.0)},
    {"cis-decalin",
     SMDSolventParameters(1.4810, 1.4788, 0.00, 0.00, 45.45, 2.2139, 0.0, 0.0)},
    {"n-decane",
     SMDSolventParameters(1.4102, 1.4094, 0.00, 0.00, 33.64, 1.9846, 0.0, 0.0)},
    {"dibromomethane", SMDSolventParameters(1.5420, 1.5389, 0.10, 0.10, 56.21,
                                            7.2273, 0.0, 0.667)},
    {"butylether",
     SMDSolventParameters(1.3992, 1.3968, 0.00, 0.45, 35.98, 3.0473, 0.0, 0.0)},
    {"o-dichlorobenzene", SMDSolventParameters(1.5515, 1.5491, 0.00, 0.04,
                                               52.72, 9.9949, 0.75, 0.25)},
    {"E-1,2-dichloroethene",
     SMDSolventParameters(1.4454, 1.4435, 0.09, 0.05, 37.13, 2.14, 0.0, 0.5)},
    {"Z-1,2-dichloroethene",
     SMDSolventParameters(1.4490, 1.4461, 0.11, 0.05, 39.8, 9.2, 0.0, 0.5)},
    {"dichloromethane",
     SMDSolventParameters(1.4242, 1.4212, 0.10, 0.05, 39.15, 8.93, 0.0, 0.667)},
    {"diethylether",
     SMDSolventParameters(1.3526, 1.3496, 0.00, 0.41, 23.96, 4.2400, 0.0, 0.0)},
    {"diethylsulfide",
     SMDSolventParameters(1.4430, 1.4401, 0.00, 0.32, 35.36, 5.723, 0.0, 0.0)},
    {"diethylamine",
     SMDSolventParameters(1.3864, 1.3825, 0.08, 0.69, 28.57, 3.5766, 0.0, 0.0)},
    {"diiodomethane",
     SMDSolventParameters(1.7425, 1.738, 0.05, 0.23, 95.25, 5.32, 0.0, 0.0)},
    {"diisopropyl ether",
     SMDSolventParameters(1.3679, 1.3653, 0.00, 0.41, 24.86, 3.38, 0.0, 0.0)},
    {"cis-1,2-dimethylcyclohexane",
     SMDSolventParameters(1.4360, 1.4336, 0.00, 0.00, 36.28, 2.06, 0.0, 0.0)},
    {"dimethyldisulfide",
     SMDSolventParameters(1.5289, 1.522, 0.00, 0.28, 48.06, 9.6, 0.0, 0.0)},
    {"N,N-dimethylacetamide",
     SMDSolventParameters(1.4380, 1.4358, 0.00, 0.78, 47.62, 37.781, 0.0, 0.0)},
    {"N,N-dimethylformamide",
     SMDSolventParameters(1.4305, 1.4280, 0.00, 0.74, 49.56, 37.219, 0.0, 0.0)},
    {"dimethylsulfoxide",
     SMDSolventParameters(1.4783, 1.4783, 0.00, 0.88, 61.78, 46.826, 0.0, 0.0)},
    {"diphenylether",
     SMDSolventParameters(1.5787, -1, 0.00, 0.20, 38.5, 3.73, 0.923, 0.0)},
    {"dipropylamine",
     SMDSolventParameters(1.4050, 1.4018, 0.08, 0.69, 32.11, 2.9112, 0.0, 0.0)},
    {"n-dodecane",
     SMDSolventParameters(1.4216, 1.4151, 0.00, 0.00, 35.85, 2.0060, 0.0, 0.0)},
    {"ethanethiol",
     SMDSolventParameters(1.4310, 1.4278, 0.00, 0.24, 33.22, 6.667, 0.0, 0.0)},
    {"ethanol",
     SMDSolventParameters(1.3611, 1.3593, 0.37, 0.48, 31.62, 24.852, 0.0, 0.0)},
    {"ethylethanoate",
     SMDSolventParameters(1.3723, 1.3704, 0.00, 0.45, 33.67, 5.9867, 0.0, 0.0)},
    {"ethylmethanoate",
     SMDSolventParameters(1.3599, 1.3575, 0.00, 0.38, 33.36, 8.3310, 0.0, 0.0)},
    {"ethylphenylether", SMDSolventParameters(1.5076, 1.5254, 0.00, 0.32, 46.65,
                                              4.1797, 0.667, 0.0)},
    {"ethylbenzene", SMDSolventParameters(1.4959, 1.4932, 0.00, 0.15, 41.38,
                                          2.4339, 0.75, 0.0)},
    {"fluorobenzene", SMDSolventParameters(1.4684, 1.4629, 0.00, 0.10, 38.37,
                                           5.42, 0.857, 0.143)},
    {"formamide",
     SMDSolventParameters(1.4472, 1.4468, 0.62, 0.60, 82.08, 108.94, 0.0, 0.0)},
    {"formicacid",
     SMDSolventParameters(1.3714, 1.3693, 0.75, 0.38, 53.44, 51.1, 0.0, 0.0)},
    {"n-heptane",
     SMDSolventParameters(1.3878, 1.3855, 0.00, 0.00, 28.28, 1.9113, 0.0, 0.0)},
    {"n-hexadecane",
     SMDSolventParameters(1.4345, 1.4325, 0.00, 0.00, 38.93, 2.0402, 0.0, 0.0)},
    {"n-hexane",
     SMDSolventParameters(1.3749, 1.3722, 0.00, 0.00, 25.75, 1.8819, 0.0, 0.0)},
    {"hexanoicacid",
     SMDSolventParameters(1.4163, 1.4146, 0.60, 0.45, 39.65, 2.6, 0.0, 0.0)},
    {"iodobenzene", SMDSolventParameters(1.6200, 1.6172, 0.00, 0.12, 55.72,
                                         4.5470, 0.857, 0.0)},
    {"iodoethane",
     SMDSolventParameters(1.5133, 1.5100, 0.00, 0.15, 40.96, 7.6177, 0.0, 0.0)},
    {"iodomethane",
     SMDSolventParameters(1.5380, 1.5270, 0.00, 0.13, 43.67, 6.8650, 0.0, 0.0)},
    {"isopropylbenzene", SMDSolventParameters(1.4915, 1.4889, 0.00, 0.16, 39.85,
                                              2.3712, 0.667, 0.0)},
    {"p-isopropyltoluene", SMDSolventParameters(1.4909, 1.4885, 0.00, 0.19,
                                                38.34, 2.2322, 0.600, 0.0)},
    {"mesitylene", SMDSolventParameters(1.4994, 1.4968, 0.00, 0.19, 39.65,
                                        2.2650, 0.667, 0.0)},
    {"methanol",
     SMDSolventParameters(1.3288, 1.3265, 0.43, 0.47, 31.77, 32.613, 0.0, 0.0)},
    {"methylbenzoate", SMDSolventParameters(1.5164, 1.5146, 0.00, 0.46, 53.5,
                                            6.7367, 0.600, 0.0)},
    {"methylbutanoate",
     SMDSolventParameters(1.3878, 1.3847, 0.00, 0.45, 35.44, 5.5607, 0.0, 0.0)},
    {"methylethanoate",
     SMDSolventParameters(1.3614, 1.3589, 0.00, 0.45, 35.59, 6.8615, 0.0, 0.0)},
    {"methylmethanoate",
     SMDSolventParameters(1.3433, 1.3415, 0.00, 0.38, 35.06, 8.8377, 0.0, 0.0)},
    {"methylpropanoate",
     SMDSolventParameters(1.3775, 1.3742, 0.00, 0.45, 35.18, 6.0777, 0.0, 0.0)},
    {"N-methylaniline", SMDSolventParameters(1.5684, 1.5681, 0.17, 0.43, 53.11,
                                             5.9600, 0.75, 0.0)},
    {"methylcyclohexane",
     SMDSolventParameters(1.4231, 1.4206, 0.00, 0.00, 33.52, 2.024, 0.0, 0.0)},
    {"N-methylformamide(E-Zmixture)",
     SMDSolventParameters(1.4319, 1.4310, 0.40, 0.55, 55.44, 181.56, 0.0, 0.0)},
    {"nitrobenzene", SMDSolventParameters(1.5562, 1.5030, 0.00, 0.28, 57.54,
                                          34.809, 0.667, 0.0)},
    {"nitroethane",
     SMDSolventParameters(1.3917, 1.3897, 0.02, 0.33, 46.25, 28.29, 0.0, 0.0)},
    {"nitromethane",
     SMDSolventParameters(1.3817, 1.3796, 0.06, 0.31, 52.58, 36.562, 0.0, 0.0)},
    {"o-nitrotoluene",
     SMDSolventParameters(1.5450, 1.5474, 0.0, 0.27, 59.12, 25.669, 0.6, 0.0)},
    {"n-nonane",
     SMDSolventParameters(1.4054, 1.4031, 0.0, 0.0, 32.21, 1.9605, 0.0, 0.0)},
    {"n-octane",
     SMDSolventParameters(1.3974, 1.3953, 0.0, 0.0, 30.43, 1.9406, 0.0, 0.0)},
    {"n-pentadecane",
     SMDSolventParameters(1.4315, 1.4298, 0.0, 0.0, 38.34, 2.0333, 0.0, 0.0)},
    {"pentanal",
     SMDSolventParameters(1.3944, 1.3917, 0.0, 0.4, 36.62, 10.0, 0.0, 0.0)},
    {"n-pentane",
     SMDSolventParameters(1.3575, 1.3547, 0.0, 0.0, 22.3, 1.8371, 0.0, 0.0)},
    {"pentanoic acid",
     SMDSolventParameters(1.4085, 1.4060, 0.60, 0.45, 38.4, 2.6924, 0.0, 0.0)},
    {"pentyl ethanoate",
     SMDSolventParameters(1.4023, -1.0, 0.0, 0.45, 36.23, 4.7297, 0.0, 0.0)},
    {"pentylamine",
     SMDSolventParameters(1.448, 1.4093, 0.16, 0.61, 35.54, 4.2010, 0.0, 0.0)},
    {"perfluorobenzene",
     SMDSolventParameters(1.3777, 1.3761, 0.00, 0.00, 31.74, 2.029, 0.5, 0.5)},
    {"propanal",
     SMDSolventParameters(1.3636, 1.3593, 0.00, 0.45, 32.48, 18.5, 0.0, 0.0)},
    {"propanoic acid",
     SMDSolventParameters(1.3869, 1.3848, 0.60, 0.45, 37.71, 3.44, 0.0, 0.0)},
    {"propanonitrile",
     SMDSolventParameters(1.3655, 1.3633, 0.02, 0.36, 38.5, 29.324, 0.0, 0.0)},
    {"propyl ethanoate",
     SMDSolventParameters(1.3842, 1.3822, 0.0, 0.45, 34.26, 5.5205, 0.0, 0.0)},
    {"propylamine",
     SMDSolventParameters(1.3870, 1.3851, 0.16, 0.61, 31.31, 4.9912, 0.0, 0.0)},
    {"pyridine", SMDSolventParameters(1.5095, 1.5073, 0.0, 0.52, 52.62, 12.978,
                                      0.833, 0.0)},
    {"tetrachloroethene",
     SMDSolventParameters(1.5053, 1.5055, 0.0, 0.0, 45.19, 2.268, 0.0, 0.667)},
    {"tetrahydrofuran",
     SMDSolventParameters(1.4050, 1.4044, 0.0, 0.48, 39.44, 7.4257, 0.0, 0.0)},
    {"tetrahydrothiophene-S,S-dioxide",
     SMDSolventParameters(1.4833, -1.0, 0.0, 0.88, 87.49, 43.962, 0.0, 0.0)},
    {"tetralin",
     SMDSolventParameters(1.5413, 1.5392, 0.0, 0.19, 47.74, 2.771, 0.6, 0.0)},
    {"thiophene",
     SMDSolventParameters(1.5289, 1.5268, 0.0, 0.15, 44.16, 2.7270, 0.8, 0.0)},
    {"thiophenol", SMDSolventParameters(1.5893, 1.580, 0.09, 0.16, 55.24,
                                        4.2728, 0.857, 0.0)},
    {"toluene",
     SMDSolventParameters(1.4961, 1.4936, 0.0, 0.14, 40.2, 2.3741, 0.857, 0.0)},
    {"trans-decalin",
     SMDSolventParameters(1.4695, 1.4671, 0.0, 0.0, 42.19, 2.1781, 0.0, 0.0)},
    {"tributylphosphate",
     SMDSolventParameters(1.4224, 1.4215, 0.0, 1.21, 27.55, 8.1781, 0.0, 0.0)},
    {"trichloroethene",
     SMDSolventParameters(1.4773, 1.4556, 0.08, 0.03, 41.45, 3.422, 0.0, 0.6)},
    {"triethylamine",
     SMDSolventParameters(1.4010, 1.3980, 0.0, 0.79, 29.1, 2.3832, 0.0, 0.0)},
    {"n-undecane",
     SMDSolventParameters(1.4398, 1.4151, 0.0, 0.0, 34.85, 1.991, 0.0, 0.0)},
    {"water", SMDSolventParameters(1.3328, 1.3323, -1.0, -1.0, -1.0, 78.355,
                                   -1.0, -1.0, true)},
    {"xylene (mixture)",
     SMDSolventParameters(1.4995, 1.4969, 0.0, 0.16, 41.38, 2.3879, 0.75, 0.0)},
    {"m-xylene",
     SMDSolventParameters(1.4972, 1.4946, 0.0, 0.16, 40.98, 2.3478, 0.75, 0.0)},
    {"o-xylene",
     SMDSolventParameters(1.5055, 1.5029, 0.0, 0.16, 42.83, 2.5454, 0.75, 0.0)},
    {"p-xylene",
     SMDSolventParameters(1.4958, 1.4932, 0.0, 0.16, 40.32, 2.2705, 0.75, 0.0)},
};

void load_solvent_parameters() {
  if (impl::solvent_parameter_data_initialized)
    return;
  impl::solvent_parameter_data_initialized = true;

  std::string data_path = solvent_data_path();

  std::string smd_filepath = "smd.json";
  if (!fs::exists(smd_filepath)) {
    smd_filepath = data_path + "/" + smd_filepath;
  }

  if (!fs::exists(smd_filepath)) {
    occ::log::debug("Skip loading SMD parameters from {}: file does not exist",
                    smd_filepath);
  } else {
    nlohmann::json json_data;
    occ::log::debug("Loading SMD parameters from {}", smd_filepath);
    std::ifstream input(smd_filepath);
    input >> json_data;
    for (const auto &j : json_data.items()) {
      smd_solvent_parameters[j.key()] = j.value();
    }
  }
}

nlohmann::json load_draco_parameters() {
  std::string data_path = solvent_data_path();

  std::string draco_filepath = "draco.json";
  if (!fs::exists(draco_filepath)) {
    draco_filepath = data_path + "/" + draco_filepath;
  }

  if (!fs::exists(draco_filepath)) {
    occ::log::debug(
        "Skip loading DRACO parameters from {}: file does not exist",
        draco_filepath);
  } else {
    nlohmann::json json_data;
    occ::log::debug("Loading DRACO parameters from {}", draco_filepath);
    std::ifstream input(draco_filepath);
    input >> json_data;
    return json_data;
  }
  return {};
}

double get_dielectric(const std::string &name) {
  const auto kv = dielectric_constant.find(name);
  if (kv == dielectric_constant.end()) {
    throw std::runtime_error(fmt::format(
        "Unknown solvent name for dielectric constant: '{}'", name));
  }
  return kv->second;
}

SMDSolventParameters get_smd_parameters(const std::string &name) {
  load_solvent_parameters();
  const auto kv = smd_solvent_parameters.find(name);
  if (kv == smd_solvent_parameters.end()) {
    throw std::runtime_error(
        fmt::format("Unknown SMD solvent name: '{}'", name));
  }
  return kv->second;
}

void list_available_solvents() {
  occ::log::info("{: <32s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} "
                 "{:>10s} {:>10s}",
                 "Solvent", "n (293K)", "acidity", "basicity", "gamma",
                 "dielectric", "aromatic", "%F,Cl,Br");
  occ::log::info("{:-<110s}", "");
  for (const auto &solvent : occ::solvent::smd_solvent_parameters) {
    const auto &param = solvent.second;
    occ::log::info("{:<32s} {:10.4f} {:10.4f} {:10.4f} {:10.4f} "
                   "{:10.4f} {:10.4f} {:10.4f}",
                   solvent.first, param.refractive_index_293K, param.acidity,
                   param.basicity, param.gamma, param.dielectric,
                   param.aromaticity, param.electronegative_halogenicity);
  }
}

} // namespace occ::solvent
