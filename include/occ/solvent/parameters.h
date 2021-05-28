#pragma once
#include <occ/3rdparty/robin_hood.h>

namespace occ::solvent
{

static inline robin_hood::unordered_map<std::string, double> dielectric_constant{
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
   {"decalin (cis/trans mixture)", 2.196},
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
   {"N-methylformamide (E/Z mixture)", 181.56},
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
   {"pentanal", 10.000},
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

}
