#pragma once
#include <fmt/core.h>
#include <memory>
#include <occ/core/linear_algebra.h>
#include <occ/dft/range_separated_parameters.h>
#include <occ/qm/spinorbital.h>
#include <string>

extern "C" {
#include <xc.h>
}

namespace occ::dft {

using occ::Array;
using occ::IVec;
using occ::Vec;
using occ::qm::SpinorbitalKind;

class DensityFunctional {
public:
  enum Identifier {
    // TODO go through and remove the deprecated functionals!
    lda_x = XC_LDA_X,
    lda_c_wigner = XC_LDA_C_WIGNER,
    lda_c_rpa = XC_LDA_C_RPA,
    lda_c_hl = XC_LDA_C_HL,
    lda_c_gl = XC_LDA_C_GL,
    lda_c_xalpha = XC_LDA_C_XALPHA,
    lda_c_vwn = XC_LDA_C_VWN,
    lda_c_vwn_rpa = XC_LDA_C_VWN_RPA,
    lda_c_pz = XC_LDA_C_PZ,
    lda_c_pz_mod = XC_LDA_C_PZ_MOD,
    lda_c_ob_pz = XC_LDA_C_OB_PZ,
    lda_c_pw = XC_LDA_C_PW,
    lda_c_pw_mod = XC_LDA_C_PW_MOD,
    lda_c_ob_pw = XC_LDA_C_OB_PW,
    lda_c_2d_amgb = XC_LDA_C_2D_AMGB,
    lda_c_2d_prm = XC_LDA_C_2D_PRM,
    lda_c_vbh = XC_LDA_C_vBH,
    lda_c_1d_csc = XC_LDA_C_1D_CSC,
    lda_x_2d = XC_LDA_X_2D,
    lda_xc_teter93 = XC_LDA_XC_TETER93,
    lda_x_1d = XC_LDA_X_1D,
    lda_c_ml1 = XC_LDA_C_ML1,
    lda_c_ml2 = XC_LDA_C_ML2,
    lda_c_gombas = XC_LDA_C_GOMBAS,
    lda_c_pw_rpa = XC_LDA_C_PW_RPA,
    lda_c_1d_loos = XC_LDA_C_1D_LOOS,
    lda_c_rc04 = XC_LDA_C_RC04,
    lda_c_vwn_1 = XC_LDA_C_VWN_1,
    lda_c_vwn_2 = XC_LDA_C_VWN_2,
    lda_c_vwn_3 = XC_LDA_C_VWN_3,
    lda_c_vwn_4 = XC_LDA_C_VWN_4,
    lda_xc_zlp = XC_LDA_XC_ZLP,
    lda_k_tf = XC_LDA_K_TF,
    lda_k_lp = XC_LDA_K_LP,
    lda_xc_ksdt = XC_LDA_XC_KSDT,
    gga_x_gam = XC_GGA_X_GAM,
    gga_c_gam = XC_GGA_C_GAM,
    gga_x_hcth_a = XC_GGA_X_HCTH_A,
    gga_x_ev93 = XC_GGA_X_EV93,
    gga_x_bgcp = XC_GGA_X_BGCP,
    gga_c_bgcp = XC_GGA_C_BGCP,
    gga_x_lambda_oc2_n = XC_GGA_X_LAMBDA_OC2_N,
    gga_x_b86_r = XC_GGA_X_B86_R,
    gga_x_lambda_ch_n = XC_GGA_X_LAMBDA_CH_N,
    gga_x_lambda_lo_n = XC_GGA_X_LAMBDA_LO_N,
    gga_x_hjs_b88_v2 = XC_GGA_X_HJS_B88_V2,
    gga_c_q2d = XC_GGA_C_Q2D,
    gga_x_q2d = XC_GGA_X_Q2D,
    gga_x_pbe_mol = XC_GGA_X_PBE_MOL,
    gga_k_tfvw = XC_GGA_K_TFVW,
    gga_k_revapbeint = XC_GGA_K_REVAPBEINT,
    gga_k_apbeint = XC_GGA_K_APBEINT,
    gga_k_revapbe = XC_GGA_K_REVAPBE,
    gga_x_ak13 = XC_GGA_X_AK13,
    gga_k_meyer = XC_GGA_K_MEYER,
    gga_x_lv_rpw86 = XC_GGA_X_LV_RPW86,
    gga_x_pbe_tca = XC_GGA_X_PBE_TCA,
    gga_x_pbeint = XC_GGA_X_PBEINT,
    gga_c_zpbeint = XC_GGA_C_ZPBEINT,
    gga_c_pbeint = XC_GGA_C_PBEINT,
    gga_c_zpbesol = XC_GGA_C_ZPBESOL,
    gga_xc_opbe_d = XC_GGA_XC_OPBE_D,
    gga_xc_opwlyp_d = XC_GGA_XC_OPWLYP_D,
    gga_xc_oblyp_d = XC_GGA_XC_OBLYP_D,
    gga_x_vmt84_ge = XC_GGA_X_VMT84_GE,
    gga_x_vmt84_pbe = XC_GGA_X_VMT84_PBE,
    gga_x_vmt_ge = XC_GGA_X_VMT_GE,
    gga_x_vmt_pbe = XC_GGA_X_VMT_PBE,
    gga_c_n12_sx = XC_GGA_C_N12_SX,
    gga_c_n12 = XC_GGA_C_N12,
    gga_x_n12 = XC_GGA_X_N12,
    gga_c_regtpss = XC_GGA_C_REGTPSS,
    gga_c_op_xalpha = XC_GGA_C_OP_XALPHA,
    gga_c_op_g96 = XC_GGA_C_OP_G96,
    gga_c_op_pbe = XC_GGA_C_OP_PBE,
    gga_c_op_b88 = XC_GGA_C_OP_B88,
    gga_c_ft97 = XC_GGA_C_FT97,
    gga_c_spbe = XC_GGA_C_SPBE,
    gga_x_ssb_sw = XC_GGA_X_SSB_SW,
    gga_x_ssb = XC_GGA_X_SSB,
    gga_x_ssb_d = XC_GGA_X_SSB_D,
    gga_xc_hcth_407p = XC_GGA_XC_HCTH_407P,
    gga_xc_hcth_p76 = XC_GGA_XC_HCTH_P76,
    gga_xc_hcth_p14 = XC_GGA_XC_HCTH_P14,
    gga_xc_b97_gga1 = XC_GGA_XC_B97_GGA1,
    gga_c_hcth_a = XC_GGA_C_HCTH_A,
    gga_x_bpccac = XC_GGA_X_BPCCAC,
    gga_c_revtca = XC_GGA_C_REVTCA,
    gga_c_tca = XC_GGA_C_TCA,
    gga_x_pbe = XC_GGA_X_PBE,
    gga_x_pbe_r = XC_GGA_X_PBE_R,
    gga_x_b86 = XC_GGA_X_B86,
    gga_x_herman = XC_GGA_X_HERMAN,
    gga_x_b86_mgc = XC_GGA_X_B86_MGC,
    gga_x_b88 = XC_GGA_X_B88,
    gga_x_g96 = XC_GGA_X_G96,
    gga_x_pw86 = XC_GGA_X_PW86,
    gga_x_pw91 = XC_GGA_X_PW91,
    gga_x_optx = XC_GGA_X_OPTX,
    gga_x_dk87_r1 = XC_GGA_X_DK87_R1,
    gga_x_dk87_r2 = XC_GGA_X_DK87_R2,
    gga_x_lg93 = XC_GGA_X_LG93,
    gga_x_ft97_a = XC_GGA_X_FT97_A,
    gga_x_ft97_b = XC_GGA_X_FT97_B,
    gga_x_pbe_sol = XC_GGA_X_PBE_SOL,
    gga_x_rpbe = XC_GGA_X_RPBE,
    gga_x_wc = XC_GGA_X_WC,
    gga_x_mpw91 = XC_GGA_X_MPW91,
    gga_x_am05 = XC_GGA_X_AM05,
    gga_x_pbea = XC_GGA_X_PBEA,
    gga_x_mpbe = XC_GGA_X_MPBE,
    gga_x_xpbe = XC_GGA_X_XPBE,
    gga_x_2d_b86_mgc = XC_GGA_X_2D_B86_MGC,
    gga_x_bayesian = XC_GGA_X_BAYESIAN,
    gga_x_pbe_jsjr = XC_GGA_X_PBE_JSJR,
    gga_x_2d_b88 = XC_GGA_X_2D_B88,
    gga_x_2d_b86 = XC_GGA_X_2D_B86,
    gga_x_2d_pbe = XC_GGA_X_2D_PBE,
    gga_c_pbe = XC_GGA_C_PBE,
    gga_c_lyp = XC_GGA_C_LYP,
    gga_c_p86 = XC_GGA_C_P86,
    gga_c_pbe_sol = XC_GGA_C_PBE_SOL,
    gga_c_pw91 = XC_GGA_C_PW91,
    gga_c_am05 = XC_GGA_C_AM05,
    gga_c_xpbe = XC_GGA_C_XPBE,
    gga_c_lm = XC_GGA_C_LM,
    gga_c_pbe_jrgx = XC_GGA_C_PBE_JRGX,
    gga_x_optb88_vdw = XC_GGA_X_OPTB88_VDW,
    gga_x_pbek1_vdw = XC_GGA_X_PBEK1_VDW,
    gga_x_optpbe_vdw = XC_GGA_X_OPTPBE_VDW,
    gga_x_rge2 = XC_GGA_X_RGE2,
    gga_c_rge2 = XC_GGA_C_RGE2,
    gga_x_rpw86 = XC_GGA_X_RPW86,
    gga_x_kt1 = XC_GGA_X_KT1,
    gga_xc_kt2 = XC_GGA_XC_KT2,
    gga_c_wl = XC_GGA_C_WL,
    gga_c_wi = XC_GGA_C_WI,
    gga_x_mb88 = XC_GGA_X_MB88,
    gga_x_sogga = XC_GGA_X_SOGGA,
    gga_x_sogga11 = XC_GGA_X_SOGGA11,
    gga_c_sogga11 = XC_GGA_C_SOGGA11,
    gga_c_wi0 = XC_GGA_C_WI0,
    gga_xc_th1 = XC_GGA_XC_TH1,
    gga_xc_th2 = XC_GGA_XC_TH2,
    gga_xc_th3 = XC_GGA_XC_TH3,
    gga_xc_th4 = XC_GGA_XC_TH4,
    gga_x_c09x = XC_GGA_X_C09X,
    gga_c_sogga11_x = XC_GGA_C_SOGGA11_X,
    gga_x_lb = XC_GGA_X_LB,
    gga_xc_hcth_93 = XC_GGA_XC_HCTH_93,
    gga_xc_hcth_120 = XC_GGA_XC_HCTH_120,
    gga_xc_hcth_147 = XC_GGA_XC_HCTH_147,
    gga_xc_hcth_407 = XC_GGA_XC_HCTH_407,
    gga_xc_edf1 = XC_GGA_XC_EDF1,
    gga_xc_xlyp = XC_GGA_XC_XLYP,
    gga_xc_b97_d = XC_GGA_XC_B97_D,
    gga_xc_pbe1w = XC_GGA_XC_PBE1W,
    gga_xc_mpwlyp1w = XC_GGA_XC_MPWLYP1W,
    gga_xc_pbelyp1w = XC_GGA_XC_PBELYP1W,
    gga_x_lbm = XC_GGA_X_LBM,
    gga_x_ol2 = XC_GGA_X_OL2,
    gga_x_apbe = XC_GGA_X_APBE,
    gga_k_apbe = XC_GGA_K_APBE,
    gga_c_apbe = XC_GGA_C_APBE,
    gga_k_tw1 = XC_GGA_K_TW1,
    gga_k_tw2 = XC_GGA_K_TW2,
    gga_k_tw3 = XC_GGA_K_TW3,
    gga_k_tw4 = XC_GGA_K_TW4,
    gga_x_htbs = XC_GGA_X_HTBS,
    gga_x_airy = XC_GGA_X_AIRY,
    gga_x_lag = XC_GGA_X_LAG,
    gga_xc_mohlyp = XC_GGA_XC_MOHLYP,
    gga_xc_mohlyp2 = XC_GGA_XC_MOHLYP2,
    gga_xc_th_fl = XC_GGA_XC_TH_FL,
    gga_xc_th_fc = XC_GGA_XC_TH_FC,
    gga_xc_th_fcfo = XC_GGA_XC_TH_FCFO,
    gga_xc_th_fco = XC_GGA_XC_TH_FCO,
    gga_c_optc = XC_GGA_C_OPTC,
    gga_c_pbeloc = XC_GGA_C_PBELOC,
    gga_xc_vv10 = XC_GGA_XC_VV10,
    gga_c_pbefe = XC_GGA_C_PBEFE,
    gga_c_op_pw91 = XC_GGA_C_OP_PW91,
    gga_x_pbefe = XC_GGA_X_PBEFE,
    gga_x_cap = XC_GGA_X_CAP,
    gga_k_vw = XC_GGA_K_VW,
    gga_k_ge2 = XC_GGA_K_GE2,
    gga_k_golden = XC_GGA_K_GOLDEN,
    gga_k_yt65 = XC_GGA_K_YT65,
    gga_k_baltin = XC_GGA_K_BALTIN,
    gga_k_lieb = XC_GGA_K_LIEB,
    gga_k_absp1 = XC_GGA_K_ABSP1,
    gga_k_absp2 = XC_GGA_K_ABSP2,
    gga_k_gr = XC_GGA_K_GR,
    gga_k_ludena = XC_GGA_K_LUDENA,
    gga_k_gp85 = XC_GGA_K_GP85,
    gga_k_pearson = XC_GGA_K_PEARSON,
    gga_k_ol1 = XC_GGA_K_OL1,
    gga_k_ol2 = XC_GGA_K_OL2,
    gga_k_fr_b88 = XC_GGA_K_FR_B88,
    gga_k_fr_pw86 = XC_GGA_K_FR_PW86,
    gga_k_dk = XC_GGA_K_DK,
    gga_k_perdew = XC_GGA_K_PERDEW,
    gga_k_vsk = XC_GGA_K_VSK,
    gga_k_vjks = XC_GGA_K_VJKS,
    gga_k_ernzerhof = XC_GGA_K_ERNZERHOF,
    gga_k_lc94 = XC_GGA_K_LC94,
    gga_k_llp = XC_GGA_K_LLP,
    gga_k_thakkar = XC_GGA_K_THAKKAR,
    gga_x_wpbeh = XC_GGA_X_WPBEH,
    gga_x_hjs_pbe = XC_GGA_X_HJS_PBE,
    gga_x_hjs_pbe_sol = XC_GGA_X_HJS_PBE_SOL,
    gga_x_hjs_b88 = XC_GGA_X_HJS_B88,
    gga_x_hjs_b97x = XC_GGA_X_HJS_B97X,
    gga_x_ityh = XC_GGA_X_ITYH,
    gga_x_sfat = XC_GGA_X_SFAT,
    hyb_gga_x_n12_sx = XC_HYB_GGA_X_N12_SX,
    hyb_gga_xc_b97_1p = XC_HYB_GGA_XC_B97_1p,
    hyb_gga_xc_b3pw91 = XC_HYB_GGA_XC_B3PW91,
    hyb_gga_xc_b3lyp = XC_HYB_GGA_XC_B3LYP,
    hyb_gga_xc_b3p86 = XC_HYB_GGA_XC_B3P86,
    hyb_gga_xc_o3lyp = XC_HYB_GGA_XC_O3LYP,
    hyb_gga_xc_mpw1k = XC_HYB_GGA_XC_mPW1K,
    hyb_gga_xc_pbeh = XC_HYB_GGA_XC_PBEH,
    hyb_gga_xc_b97 = XC_HYB_GGA_XC_B97,
    hyb_gga_xc_b97_1 = XC_HYB_GGA_XC_B97_1,
    hyb_gga_xc_b97_2 = XC_HYB_GGA_XC_B97_2,
    hyb_gga_xc_x3lyp = XC_HYB_GGA_XC_X3LYP,
    hyb_gga_xc_b1wc = XC_HYB_GGA_XC_B1WC,
    hyb_gga_xc_b97_k = XC_HYB_GGA_XC_B97_K,
    hyb_gga_xc_b97_3 = XC_HYB_GGA_XC_B97_3,
    hyb_gga_xc_mpw3pw = XC_HYB_GGA_XC_MPW3PW,
    hyb_gga_xc_b1lyp = XC_HYB_GGA_XC_B1LYP,
    hyb_gga_xc_b1pw91 = XC_HYB_GGA_XC_B1PW91,
    hyb_gga_xc_mpw1pw = XC_HYB_GGA_XC_mPW1PW,
    hyb_gga_xc_mpw3lyp = XC_HYB_GGA_XC_MPW3LYP,
    hyb_gga_xc_sb98_1a = XC_HYB_GGA_XC_SB98_1a,
    hyb_gga_xc_sb98_1b = XC_HYB_GGA_XC_SB98_1b,
    hyb_gga_xc_sb98_1c = XC_HYB_GGA_XC_SB98_1c,
    hyb_gga_xc_sb98_2a = XC_HYB_GGA_XC_SB98_2a,
    hyb_gga_xc_sb98_2b = XC_HYB_GGA_XC_SB98_2b,
    hyb_gga_xc_sb98_2c = XC_HYB_GGA_XC_SB98_2c,
    hyb_gga_x_sogga11_x = XC_HYB_GGA_X_SOGGA11_X,
    hyb_gga_xc_hse03 = XC_HYB_GGA_XC_HSE03,
    hyb_gga_xc_hse06 = XC_HYB_GGA_XC_HSE06,
    hyb_gga_xc_hjs_pbe = XC_HYB_GGA_XC_HJS_PBE,
    hyb_gga_xc_hjs_pbe_sol = XC_HYB_GGA_XC_HJS_PBE_SOL,
    hyb_gga_xc_hjs_b88 = XC_HYB_GGA_XC_HJS_B88,
    hyb_gga_xc_hjs_b97x = XC_HYB_GGA_XC_HJS_B97X,
    hyb_gga_xc_cam_b3lyp = XC_HYB_GGA_XC_CAM_B3LYP,
    hyb_gga_xc_tuned_cam_b3lyp = XC_HYB_GGA_XC_TUNED_CAM_B3LYP,
    hyb_gga_xc_bhandh = XC_HYB_GGA_XC_BHANDH,
    hyb_gga_xc_bhandhlyp = XC_HYB_GGA_XC_BHANDHLYP,
    hyb_gga_xc_mb3lyp_rc04 = XC_HYB_GGA_XC_MB3LYP_RC04,
    hyb_gga_xc_mpwlyp1m = XC_HYB_GGA_XC_MPWLYP1M,
    hyb_gga_xc_revb3lyp = XC_HYB_GGA_XC_REVB3LYP,
    hyb_gga_xc_camy_blyp = XC_HYB_GGA_XC_CAMY_BLYP,
    hyb_gga_xc_pbe0_13 = XC_HYB_GGA_XC_PBE0_13,
    hyb_gga_xc_b3lyps = XC_HYB_GGA_XC_B3LYPs,
    hyb_gga_xc_wb97 = XC_HYB_GGA_XC_WB97,
    hyb_gga_xc_wb97x = XC_HYB_GGA_XC_WB97X,
    hyb_gga_xc_lrc_wpbeh = XC_HYB_GGA_XC_LRC_WPBEH,
    hyb_gga_xc_wb97x_v = XC_HYB_GGA_XC_WB97X_V,
    hyb_gga_xc_lcy_pbe = XC_HYB_GGA_XC_LCY_PBE,
    hyb_gga_xc_lcy_blyp = XC_HYB_GGA_XC_LCY_BLYP,
    hyb_gga_xc_lc_vv10 = XC_HYB_GGA_XC_LC_VV10,
    hyb_gga_xc_camy_b3lyp = XC_HYB_GGA_XC_CAMY_B3LYP,
    hyb_gga_xc_wb97x_d = XC_HYB_GGA_XC_WB97X_D,
    hyb_gga_xc_hpbeint = XC_HYB_GGA_XC_HPBEINT,
    hyb_gga_xc_lrc_wpbe = XC_HYB_GGA_XC_LRC_WPBE,
    hyb_gga_xc_b3lyp5 = XC_HYB_GGA_XC_B3LYP5,
    hyb_gga_xc_edf2 = XC_HYB_GGA_XC_EDF2,
    hyb_gga_xc_cap0 = XC_HYB_GGA_XC_CAP0,
    mgga_c_dldf = XC_MGGA_C_DLDF,
    mgga_xc_zlp = XC_MGGA_XC_ZLP,
    mgga_xc_otpss_d = XC_MGGA_XC_OTPSS_D,
    mgga_c_cs = XC_MGGA_C_CS,
    mgga_c_mn12_sx = XC_MGGA_C_MN12_SX,
    mgga_c_mn12_l = XC_MGGA_C_MN12_L,
    mgga_c_m11_l = XC_MGGA_C_M11_L,
    mgga_c_m11 = XC_MGGA_C_M11,
    mgga_c_m08_so = XC_MGGA_C_M08_SO,
    mgga_c_m08_hx = XC_MGGA_C_M08_HX,
    mgga_x_lta = XC_MGGA_X_LTA,
    mgga_x_tpss = XC_MGGA_X_TPSS,
    mgga_x_m06_l = XC_MGGA_X_M06_L,
    mgga_x_gvt4 = XC_MGGA_X_GVT4,
    mgga_x_tau_hcth = XC_MGGA_X_TAU_HCTH,
    mgga_x_br89 = XC_MGGA_X_BR89,
    mgga_x_bj06 = XC_MGGA_X_BJ06,
    mgga_x_tb09 = XC_MGGA_X_TB09,
    mgga_x_rpp09 = XC_MGGA_X_RPP09,
    mgga_x_2d_prhg07 = XC_MGGA_X_2D_PRHG07,
    mgga_x_2d_prhg07_prp10 = XC_MGGA_X_2D_PRHG07_PRP10,
    mgga_x_revtpss = XC_MGGA_X_REVTPSS,
    mgga_x_pkzb = XC_MGGA_X_PKZB,
    mgga_x_m05 = XC_MGGA_X_M05,
    mgga_x_m05_2x = XC_MGGA_X_M05_2X,
    mgga_x_m06_hf = XC_MGGA_X_M06_HF,
    mgga_x_m06 = XC_MGGA_X_M06,
    hyb_mgga_x_m06_2x = XC_HYB_MGGA_X_M06_2X,
    mgga_x_m08_hx = XC_MGGA_X_M08_HX,
    mgga_x_m08_so = XC_MGGA_X_M08_SO,
    mgga_x_ms0 = XC_MGGA_X_MS0,
    mgga_x_ms1 = XC_MGGA_X_MS1,
    mgga_x_ms2 = XC_MGGA_X_MS2,
    mgga_x_m11 = XC_MGGA_X_M11,
    mgga_x_m11_l = XC_MGGA_X_M11_L,
    mgga_x_mn12_l = XC_MGGA_X_MN12_L,
    mgga_c_cc06 = XC_MGGA_C_CC06,
    mgga_x_mk00 = XC_MGGA_X_MK00,
    mgga_c_tpss = XC_MGGA_C_TPSS,
    mgga_c_vsxc = XC_MGGA_C_VSXC,
    mgga_c_m06_l = XC_MGGA_C_M06_L,
    mgga_c_m06_hf = XC_MGGA_C_M06_HF,
    mgga_c_m06 = XC_MGGA_C_M06,
    mgga_c_m06_2x = XC_MGGA_C_M06_2X,
    mgga_c_m05 = XC_MGGA_C_M05,
    mgga_c_m05_2x = XC_MGGA_C_M05_2X,
    mgga_c_pkzb = XC_MGGA_C_PKZB,
    mgga_c_bc95 = XC_MGGA_C_BC95,
    mgga_c_revtpss = XC_MGGA_C_REVTPSS,
    mgga_xc_tpsslyp1w = XC_MGGA_XC_TPSSLYP1W,
    mgga_x_mk00b = XC_MGGA_X_MK00B,
    mgga_x_bloc = XC_MGGA_X_BLOC,
    mgga_x_modtpss = XC_MGGA_X_MODTPSS,
    mgga_c_tpssloc = XC_MGGA_C_TPSSLOC,
    mgga_x_mbeef = XC_MGGA_X_MBEEF,
    mgga_x_mbeefvdw = XC_MGGA_X_MBEEFVDW,
    mgga_xc_b97m_v = XC_MGGA_XC_B97M_V,
    mgga_x_mvs = XC_MGGA_X_MVS,
    mgga_x_mn15_l = XC_MGGA_X_MN15_L,
    mgga_c_mn15_l = XC_MGGA_C_MN15_L,
    mgga_x_scan = XC_MGGA_X_SCAN,
    mgga_c_scan = XC_MGGA_C_SCAN,
    mgga_c_mn15 = XC_MGGA_C_MN15,
    mgga_x_r2scan = XC_MGGA_X_R2SCAN,
    mgga_c_r2scan = XC_MGGA_C_R2SCAN,
    hyb_mgga_x_dldf = XC_HYB_MGGA_X_DLDF,
    hyb_mgga_x_ms2h = XC_HYB_MGGA_X_MS2H,
    hyb_mgga_x_mn12_sx = XC_HYB_MGGA_X_MN12_SX,
    hyb_mgga_x_scan0 = XC_HYB_MGGA_X_SCAN0,
    hyb_mgga_x_mn15 = XC_HYB_MGGA_X_MN15,
    hyb_mgga_xc_b88b95 = XC_HYB_MGGA_XC_B88B95,
    hyb_mgga_xc_b86b95 = XC_HYB_MGGA_XC_B86B95,
    hyb_mgga_xc_pw86b95 = XC_HYB_MGGA_XC_PW86B95,
    hyb_mgga_xc_bb1k = XC_HYB_MGGA_XC_BB1K,
    hyb_mgga_xc_mpw1b95 = XC_HYB_MGGA_XC_MPW1B95,
    hyb_mgga_xc_mpwb1k = XC_HYB_MGGA_XC_MPWB1K,
    hyb_mgga_xc_x1b95 = XC_HYB_MGGA_XC_X1B95,
    hyb_mgga_xc_xb1k = XC_HYB_MGGA_XC_XB1K,
    hyb_mgga_xc_pw6b95 = XC_HYB_MGGA_XC_PW6B95,
    hyb_mgga_xc_pwb6k = XC_HYB_MGGA_XC_PWB6K,
    hyb_mgga_xc_tpssh = XC_HYB_MGGA_XC_TPSSH,
    hyb_mgga_xc_revtpssh = XC_HYB_MGGA_XC_REVTPSSH,
    hyb_mgga_xc_m08_hx = XC_HYB_MGGA_XC_M08_HX,
    hyb_mgga_xc_m08_so = XC_HYB_MGGA_XC_M08_SO,
    hyb_mgga_xc_m11 = XC_HYB_MGGA_XC_M11,
    hyb_mgga_x_mvsh = XC_HYB_MGGA_X_MVSH,
    hyb_mgga_xc_wb97m_v = XC_HYB_MGGA_XC_WB97M_V
  };

  enum Family {
    LDA = XC_FAMILY_LDA,
    GGA = XC_FAMILY_GGA,
    HGGA = XC_FAMILY_HYB_GGA,
    MGGA = XC_FAMILY_MGGA,
    HMGGA = XC_FAMILY_HYB_MGGA
  };
  enum Kind {
    Exchange = XC_EXCHANGE,
    Correlation = XC_CORRELATION,
    ExchangeCorrelation = XC_EXCHANGE_CORRELATION,
    Kinetic = XC_KINETIC
  };

  struct Result {
    Result(size_t npt, Family family, SpinorbitalKind kind) : npts{npt} {
      if (kind == SpinorbitalKind::Restricted) {
        exc = Vec::Zero(npt);
        vrho = MatRM::Zero(npt, 1);
        if (family == GGA || family == HGGA) {
          vsigma = MatRM::Zero(npt, 1);
          have_vsigma = true;
        } else if (family == MGGA || family == HMGGA) {
          vsigma = MatRM::Zero(npt, 1);
          have_vsigma = true;
          vlaplacian = MatRM::Zero(npt, 1);
          vtau = MatRM::Zero(npt, 1);
          have_vtau = true;
        }
      } else {
        exc = Vec::Zero(npt);
        vrho = MatRM::Zero(npt, 2);
        if (family == GGA || family == HGGA) {
          vsigma = MatRM::Zero(npt, 3);
          have_vsigma = true;
        } else if (family == MGGA || family == HMGGA) {
          vsigma = MatRM::Zero(npt, 3);
          have_vsigma = true;
          vlaplacian = MatRM::Zero(npt, 2);
          vtau = MatRM::Zero(npt, 2);
          have_vtau = true;
        }
      }
    }
    size_t npts{0};
    bool have_vsigma{false};
    bool have_vtau{false};
    Vec exc;
    // must be row major for libXC interface
    MatRM vrho;
    MatRM vsigma;
    MatRM vlaplacian;
    MatRM vtau;
    Result &operator+=(const Result &right) {
      exc.array() += right.exc.array();
      vrho.array() += right.vrho.array();
      if (right.have_vsigma) {
        if (!have_vsigma)
          vsigma = right.vsigma;
        else
          vsigma.array() += right.vsigma.array();
        have_vsigma = true;
      }
      if (right.have_vtau) {
        if (!have_vtau) {
          vlaplacian = right.vlaplacian;
          vtau = right.vtau;
        } else {
          vlaplacian.array() += right.vlaplacian.array();
          vtau.array() += right.vtau.array();
        }
        have_vtau = true;
      }
      return *this;
    }
    void weight_by(const Vec &weights) {
      exc.array() *= weights.array();
      vrho.array().colwise() *= weights.array();
      if (vsigma.size() > 0) {
        vsigma.array().colwise() *= weights.array();
      }
      if (vlaplacian.size() > 0) {
        vlaplacian.array().colwise() *= weights.array();
        vtau.array().colwise() *= weights.array();
      }
    }
  };

  struct Params {
    Params(size_t npt, Family family, SpinorbitalKind kind) : npts(npt) {
      if (kind == SpinorbitalKind::Restricted) {
        rho.resize(npt, 1);
        if (family == GGA || family == HGGA) {
          sigma.resize(npt, 1);
        } else if (family == MGGA || family == HMGGA) {
          sigma.resize(npt, 1);
          laplacian.resize(npt, 1);
          tau.resize(npt, 1);
        }
      } else {
        rho.resize(npt, 2);
        if (family == GGA || family == HGGA) {
          sigma.resize(npt, 3);
        } else if (family == MGGA || family == HMGGA) {
          sigma.resize(npt, 3);
          laplacian.resize(npt, 2);
          tau.resize(npt, 2);
        }
      }
    }
    size_t npts{0};
    // must be row major for libXC interface
    MatRM rho;
    MatRM sigma;
    MatRM laplacian;
    MatRM tau;
  };

  DensityFunctional(const std::string &, bool polarized = false);
  DensityFunctional(Identifier, bool polarized = false);

  void set_name(const std::string &name) { m_func_name = name; }
  double scale_factor() const { return m_factor; }
  void set_scale_factor(double fac) { m_factor = fac; }
  void set_exchange_factor(double fac) { m_exchange_factor_override = fac; }

  bool polarized() const { return m_polarized; }
  Family family() const {
    xc_func_type func;
    int integer_id = static_cast<int>(m_func_id);
    int err = xc_func_init(&func, integer_id,
                           m_polarized ? XC_POLARIZED : XC_UNPOLARIZED);
    if (err != 0) {
      throw std::runtime_error(fmt::format(
          "Error initialiizing functional with id: {}", integer_id));
    }
    Family f = static_cast<Family>(func.info->family);
    xc_func_end(&func);
    return f;
  }
  Kind kind() const {
    xc_func_type func;

    int integer_id = static_cast<int>(m_func_id);
    int err = xc_func_init(&func, integer_id,
                           m_polarized ? XC_POLARIZED : XC_UNPOLARIZED);
    if (err != 0) {
      throw std::runtime_error(fmt::format(
          "Error initialiizing functional with id: {}", integer_id));
    }

    Kind k = static_cast<Kind>(func.info->kind);
    xc_func_end(&func);
    return k;
  }
  Identifier id() const { return static_cast<Identifier>(m_func_id); }
  const std::string &name() const { return m_func_name; }
  std::string kind_string() const {
    switch (kind()) {
    case Exchange:
      return "exchange";
    case Correlation:
      return "correlation";
    case ExchangeCorrelation:
      return "exchange-correlation";
    case Kinetic:
      return "kinetic";
    default:
      return "unknown kind";
    }
  }

  double exact_exchange_factor() const {
    if (m_exchange_factor_override != 0.0)
      return m_exchange_factor_override;
    switch (family()) {
    case HGGA:
    case HMGGA: {
      xc_func_type func;
      int integer_id = static_cast<int>(m_func_id);
      int err = xc_func_init(&func, integer_id,
                             m_polarized ? XC_POLARIZED : XC_UNPOLARIZED);
      if (err != 0) {
        throw std::runtime_error(fmt::format(
            "Error initialiizing functional with id: {}", integer_id));
      }

      double fac = xc_hyb_exx_coef(&func);
      xc_func_end(&func);
      return fac;
    }
    default:
      return 0;
    }
  }

  inline auto range_separated_parameters() const {
    RangeSeparatedParameters params;
    switch (family()) {
    case HGGA:
    case HMGGA: {
      xc_func_type func;
      int integer_id = static_cast<int>(m_func_id);
      int err = xc_func_init(&func, integer_id,
                             m_polarized ? XC_POLARIZED : XC_UNPOLARIZED);
      if (err != 0) {
        throw std::runtime_error(fmt::format(
            "Error initialiizing functional with id: {}", integer_id));
      }

      xc_hyb_cam_coef(&func, &params.omega, &params.alpha, &params.beta);
      xc_func_end(&func);
      return params;
    }
    default:
      return params;
    }
  }

  int derivative_order() const {
    switch (family()) {
    case LDA:
      return 0;
    case GGA:
    case HGGA:
      return 1;
    case MGGA:
    case HMGGA:
      return 2;
    default:
      return 0;
    }
  }

  Result evaluate(const Params &params) const;

  std::string family_string() const {
    switch (family()) {
    case LDA:
      return "LDA";
    case GGA:
      return "GGA";
    case HGGA:
      return "hybrid GGA";
    case MGGA:
      return "meta-GGA";
    case HMGGA:
      return "hybrid meta-GGA";
    default:
      return "unknown family";
    }
  }
  static int functional_id(const std::string &);

  bool needs_nlc_correction() const;

private:
  double m_exchange_factor_override{0.0};
  double m_factor{1.0};
  Identifier m_func_id;
  bool m_polarized{false};
  std::string m_func_name{"unknown"};
};

} // namespace occ::dft
