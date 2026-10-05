//--------------------------------------------------------------
// encoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _ENCODE_HPP
#define _ENCODE_HPP

#include    <mfa/mfa_data.hpp>
#include    <mfa/mfa.hpp>
#include    <mfa/decode.hpp>
#include    <mfa/new_knots.hpp>
#include    <mfa/ns_initializer_checkpoint.hpp>
#include    <mfa/ns_trace_aware_initializer.hpp>
#include    <mfa/ns_hard_trace_bounds.hpp>
#include    <mfa/ns_integral_balance.hpp>
#include    <mfa/ns_knot_span_integral_balance.hpp>
#include    <mfa/reg_mfa_tikhonov.hpp>

#include    <set>
#include    <algorithm>
#include    <fstream>
#include    <iomanip>
#include    <map>
#include    <memory>
#include    <random>
#include    <stdexcept>

//BFGS optimizer for optimizing PDE error
#include <Eigen/Core>
#include <iostream>
#include <LBFGSB.h>

using namespace LBFGSpp;
//

// temporary utilities for testing
#include    <ctime>
#include    <chrono>
#include    <mpi.h>     // for MPI_Wtime() only

#ifndef      MFA_NO_WEIGHTS
#include    "coin/ClpSimplex.hpp"
#include    "coin/ClpInterior.hpp"
#endif

template <typename T>                                       // float or double
class NewKnots;

template <typename T>                                       // float or double
struct MFA;

namespace mfa
{
  template<typename T>
  inline T readEnv(const char* varName, T defaultVal) {
    if (auto env = std::getenv(varName)) {
      std::istringstream iss(env);
      T x;
      if (iss >> x) return x;
      std::cerr << "Warning: cannot parse $" << varName
                << " (“" << env << "”), using " << defaultVal << "\n";
    }
    return defaultVal;
  }

  template<typename T>
  class ObjectiveFunction_LBFGS
{
private:
  int n;
public:
  template <typename>
  friend class NewKnots;

  template <typename>
  friend class Encoder;
  using VectorType = Eigen::Matrix<T,Eigen::Dynamic,1>;
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    double lambda;
    double nu=0.001;
  T lambda_data_u = readEnv<T>("MFA_LAMBDA_DATA_U", T(1.0));
  T lambda_data_v = readEnv<T>("MFA_LAMBDA_DATA_V", T(1.0));
  T lambda_data_p = readEnv<T>("MFA_LAMBDA_DATA_P", T(1.0));

  T lambda_xmom   = readEnv<T>("MFA_LAMBDA_XMOM",   T(100.0));
  T lambda_ymom   = readEnv<T>("MFA_LAMBDA_YMOM",   T(100.0));
  T lambda_div    = readEnv<T>("MFA_LAMBDA_DIV",    T(1.0));

  T lambda_ic_u   = readEnv<T>("MFA_LAMBDA_IC_U",   T(1.0));
  T lambda_ic_v   = readEnv<T>("MFA_LAMBDA_IC_V",   T(1.0));
  T lambda_ic_p   = readEnv<T>("MFA_LAMBDA_IC_P",   T(1.0));

  T lambda_bc_u   = readEnv<T>("MFA_LAMBDA_BC_U",   T(1.0));
  T lambda_bc_v   = readEnv<T>("MFA_LAMBDA_BC_V",   T(1.0));
  T lambda_bc_p   = readEnv<T>("MFA_LAMBDA_BC_P",   T(1.0));
  T lambda_gauge  = readEnv<T>("MFA_LAMBDA_GAUGE",  T(1.0));

    NSIntegralBalanceConfig<T> integral_config;
    std::unique_ptr<NSIntegralBalanceOperators<T> > integral_operators;
    std::unique_ptr<NSIntegralBalanceOperators<T> > global_integral_operators;
    std::unique_ptr<NSKnotSpanGLBalanceOperators<T> >
      knot_span_integral_operators;
    std::unique_ptr<NSKnotSpanGLBalanceOperators<T> >
      global_knot_span_integral_operators;
    VectorX<T> bcx_physics_mask;
    VectorX<T> bcy_physics_mask;
    T active_bcx_count = T(0);
    T active_bcy_count = T(0);
    bool physics_time_mask_active = false;

    MFA_Data<T>& mfa_data;
    MFA_Data<T>& mfa_data2;
    MFA_Data<T>& mfa_data3;
    int dom_dim;
    int verbose;
    const PointSet<T>& input;
    const PointSet<T>& input_IC;
    const PointSet<T>& input_BC;
    const PointSet<T>& input_BCx;
    const PointSet<T>& input_BCy;
    size_t max_num_curves;
    SparseMatrixX<T> N1;
    SparseMatrixX<T> N1t; //transpose
    SparseMatrixX<T> N1_x;
    SparseMatrixX<T> N1t_x;
    SparseMatrixX<T> N1_xx;
    SparseMatrixX<T> N1t_xx;
    SparseMatrixX<T> N1_t; //differentiation wrt time
    SparseMatrixX<T> N1t_t;

    SparseMatrixX<T> N1_y;
    SparseMatrixX<T> N1t_y;
    SparseMatrixX<T> N1_yy;
    SparseMatrixX<T> N1t_yy;


    SparseMatrixX<T> N2;   //Ny from mfa_data2
    SparseMatrixX<T> N2t; //transpose
    SparseMatrixX<T> N2_x;
    SparseMatrixX<T> N2t_x;
    SparseMatrixX<T> N2_xx;
    SparseMatrixX<T> N2t_xx;
    SparseMatrixX<T> N2_t; //differentiation wrt time
    SparseMatrixX<T> N2t_t;

    SparseMatrixX<T> N2_y;
    SparseMatrixX<T> N2t_y;
    SparseMatrixX<T> N2_yy;
    SparseMatrixX<T> N2t_yy;

    SparseMatrixX<T> N3;   //Ny from mfa_data2
    SparseMatrixX<T> N3t; //transpose
    SparseMatrixX<T> N3_x;
    SparseMatrixX<T> N3t_x;

    SparseMatrixX<T> N3_y;
    SparseMatrixX<T> N3t_y;

    // G maps pressure values on the collocation grid to one spatial mean
    // for every time slice.  The gauge is therefore
    //     mean_{x,y} p(x,y,t) = 0  for every represented t,
    // rather than an L2 shrinkage of the complete pressure field.
    SparseMatrixX<T> G;
    SparseMatrixX<T> Gt;

    SparseMatrixX<T> B;
    SparseMatrixX<T> Bt;
    SparseMatrixX<T> Bx;
    SparseMatrixX<T> Bxt;
    SparseMatrixX<T> By;
    SparseMatrixX<T> Byt;
    SparseMatrixX<T> IC;
    SparseMatrixX<T> ICt;
    VectorX<T> Q_IC;
    VectorX<T> Q_BC;
    VectorX<T> Q_BCx;
    VectorX<T> Q_BCy;
    VectorX<T> Q;
    VectorX<T> Q_IC2;
    VectorX<T> Q_BC2;
    VectorX<T> Q_BCx2;
    VectorX<T> Q_BCy2;
    VectorX<T> Q2;

    VectorX<T> Q_IC3;
    VectorX<T> Q_BC3;
    VectorX<T> Q_BCx3;
    VectorX<T> Q_BCy3;
    VectorX<T> Q3;
    VectorX<T> data_active;
    VectorX<T> pde_active;
    VectorX<T> data_validation;
    VectorX<T> pde_validation;
    T data_active_count = T(0);
    T pde_active_count = T(0);
    T data_validation_count = T(0);
    T pde_validation_count = T(0);
    bool suppress_diagnostics{false};
    bool prefiltered_pde_benchmark =
      readEnv<int>("MFA_PREFILTERED_PDE_BENCHMARK", 0) != 0;
    std::vector<int> prefiltered_pde_rows;
    double prefiltered_construction_seconds = 0.0;
   ObjectiveFunction_LBFGS(int n_,
	      double lambda_,
	      double nu_,
	      MFA_Data<T>& mfa_data_,
	      MFA_Data<T>& mfa_data_2_,
	      MFA_Data<T>& mfa_data_3_,
	      const PointSet<T>& input_,
	      const PointSet<T>& input_IC_,
	      const PointSet<T>& input_BC_,
	      const PointSet<T>& input_BCx_,
	      const PointSet<T>& input_BCy_,
	      int verbose_)
    : n(n_),
      lambda(lambda_),
      nu(nu_),
      mfa_data(mfa_data_),
      mfa_data2(mfa_data_2_),
      mfa_data3(mfa_data_3_),
      dom_dim(mfa_data_.dom_dim),
      verbose(verbose_),
      input(input_),
      input_IC(input_IC_),
      input_BC(input_BC_),
      input_BCx(input_BCx_),
      input_BCy(input_BCy_),
      max_num_curves(1.0e5){
    integral_config =
      NSIntegralBalanceConfig<T>::fromEnvironment(static_cast<T>(nu_));
    nu = static_cast<double>(integral_config.viscosity);
    if (prefiltered_pde_benchmark)
    {
      if (integral_config.operatorsRequired())
        throw std::invalid_argument(
          "Integral balance is incompatible with MFA_PREFILTERED_PDE_BENCHMARK");
      initializePrefilteredPdeBenchmark();
    }
    else
      initializeMatrices();
  }
      ~ObjectiveFunction_LBFGS() {}
    void initializeMatrices() {
      TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[0];


      N1.resize(input.npts, t.nctrl_pts.prod());
      N1t.resize(t.nctrl_pts.prod(), input.npts);
      N1_x.resize(input.npts, t.nctrl_pts.prod());
      N1t_x.resize(t.nctrl_pts.prod(), input.npts);
      N1_xx.resize(input.npts, t.nctrl_pts.prod());
      N1t_xx.resize(t.nctrl_pts.prod(), input.npts);
      N1_y.resize(input.npts, t.nctrl_pts.prod());
      N1t_y.resize(t.nctrl_pts.prod(), input.npts);
      N1_yy.resize(input.npts, t.nctrl_pts.prod());
      N1t_yy.resize(t.nctrl_pts.prod(), input.npts);

      N1_t.resize(input.npts, t.nctrl_pts.prod());
      N1t_t.resize(t.nctrl_pts.prod(), input.npts);

      N2.resize(input.npts, t.nctrl_pts.prod());
      N2t.resize(t.nctrl_pts.prod(), input.npts);
      N2_x.resize(input.npts, t.nctrl_pts.prod());
      N2t_x.resize(t.nctrl_pts.prod(), input.npts);
      N2_xx.resize(input.npts, t.nctrl_pts.prod());
      N2t_xx.resize(t.nctrl_pts.prod(), input.npts);
      N2_y.resize(input.npts, t.nctrl_pts.prod());
      N2t_y.resize(t.nctrl_pts.prod(), input.npts);
      N2_yy.resize(input.npts, t.nctrl_pts.prod());
      N2t_yy.resize(t.nctrl_pts.prod(), input.npts);

      N2_t.resize(input.npts, t.nctrl_pts.prod());
      N2t_t.resize(t.nctrl_pts.prod(), input.npts);


      N3.resize(input.npts, t.nctrl_pts.prod());
      N3t.resize(t.nctrl_pts.prod(), input.npts);
      N3_x.resize(input.npts, t.nctrl_pts.prod());
      N3t_x.resize(t.nctrl_pts.prod(), input.npts);
      N3_y.resize(input.npts, t.nctrl_pts.prod());
      N3t_y.resize(t.nctrl_pts.prod(), input.npts);

      CollMatrixUnified(0, N1, N1t);
      CollMatrixUnified2(0, N2, N2t);
      VectorXi derivs = VectorXi::Ones(dom_dim);
      derivs(0) = 1; // derivs (0) => x , derivs (1) => y
      derivs(1) = 0;
      derivs(2) = 0;
      CollMatrixUnifiedDers(0, derivs, N1_x, N1t_x);
      CollMatrixUnifiedDers2(0, derivs, N2_x, N2t_x);
      CollMatrixUnifiedDers2(0, derivs, N3_x, N3t_x);

      derivs(0) = 2; // derivs (0) => x , derivs (1) => t
      derivs(1) = 0;
      derivs(2) = 0;
      CollMatrixUnifiedDers(0, derivs, N1_xx, N1t_xx);
      CollMatrixUnifiedDers2(0, derivs, N2_xx, N2t_xx);



      derivs(0) = 0; // derivs (0) => x , derivs (1) => y
      derivs(1) = 1;
      derivs(2) = 0;
      CollMatrixUnifiedDers(0, derivs, N1_y, N1t_y);
      CollMatrixUnifiedDers2(0, derivs, N2_y, N2t_y);
      CollMatrixUnifiedDers2(0, derivs, N3_y, N3t_y);


      derivs(0) = 0; // derivs (0) => x , derivs (1) => t
      derivs(1) = 2;
      derivs(2) = 0;
      CollMatrixUnifiedDers(0, derivs, N1_yy, N1t_yy);
      CollMatrixUnifiedDers2(0, derivs, N2_yy, N2t_yy);



      derivs(0) = 0; // derivs (0) => x , derivs (1) => t
      derivs(1) = 0;
      derivs(2) = 1;
      CollMatrixUnifiedDers(0, derivs, N1_t, N1t_t);
      CollMatrixUnifiedDers2(0, derivs, N2_t, N2t_t);
      // Scale derivatives from normalized to physical coordinates.
      const T x_extent = input.domain.col(0).maxCoeff() - input.domain.col(0).minCoeff();
      const T y_extent = input.domain.col(1).maxCoeff() - input.domain.col(1).minCoeff();
      const T t_extent = input.domain.col(2).maxCoeff() - input.domain.col(2).minCoeff();
      if (x_extent <= T(0) || y_extent <= T(0) || t_extent <= T(0))
        throw std::invalid_argument("NS objective requires positive x, y, and t extents");
      const T scalex  = T(1) / x_extent;
      const T scalexx = scalex * scalex;
      const T scaley  = T(1) / y_extent;
      const T scaleyy = scaley * scaley;
      const T scalet  = T(1) / t_extent;
      N1_x = N1_x * scalex;
      N1_xx = N1_xx * scalexx;
      N1_y = N1_y * scaley;
      N1_yy = N1_yy * scaleyy;
      N1t_x = N1t_x * scalex;
      N1t_xx = N1t_xx * scalexx;
      N1t_y = N1t_y * scaley;
      N1t_yy = N1t_yy * scaleyy;
      N1_t = N1_t * scalet;
      N1t_t = N1t_t * scalet;

      N2_x = N2_x * scalex;
      N2_xx = N2_xx * scalexx;
      N2_y = N2_y * scaley;
      N2_yy = N2_yy * scaleyy;
      N2t_x = N2t_x * scalex;
      N2t_xx = N2t_xx * scalexx;
      N2t_y = N2t_y * scaley;
      N2t_yy = N2t_yy * scaleyy;
      N2_t = N2_t * scalet;
      N2t_t = N2t_t * scalet;

      N3_x = N3_x * scalex;
      N3_y = N3_y * scaley;
      N3t_x = N3t_x * scalex;
      N3t_y = N3t_y * scaley;


      const int icline = input_IC.npts;
      int num_reg_conds = t.nctrl_pts.prod();
      IC.resize(icline, num_reg_conds);
      ICt.resize(num_reg_conds, icline);
      SparseMatrixX<T> ICt1(num_reg_conds, icline);
      ICMatrix(0, IC, ICt1);
      ICt = ICt1; // only C2 regularization


// ==========================================
      // DYNAMICALLY SIZE BOUNDARY MATRICES
      // ==========================================
      num_reg_conds = t.nctrl_pts.prod();

      // 1. X-Boundary
      int bcline_x = input_BCx.npts;
      Bx.resize(bcline_x, num_reg_conds);
      Bxt.resize(num_reg_conds, bcline_x);
      SparseMatrixX<T> Bxt1(num_reg_conds, bcline_x);
      BCxMatrix(0, Bx, Bxt1);
      Bxt = Bxt1;

      // 2. Y-Boundary
      int bcline_y = input_BCy.npts;
      By.resize(bcline_y, num_reg_conds);
      Byt.resize(num_reg_conds, bcline_y);
      SparseMatrixX<T> Byt1(num_reg_conds, bcline_y);
      BCyMatrix(0, By, Byt1);
      Byt = Byt1;


      // Initialize Q_IC
      Q_IC.resize(ICt.cols());
      Q_IC2.resize(ICt.cols());
      Q_IC3.resize(ICt.cols());

      VectorX<T> pt_coords_IC(mfa_data.dim());
      VectorX<T> pt_coords_IC2(mfa_data2.dim());
      VectorX<T> pt_coords_IC3(mfa_data3.dim());
      for (auto input_it_IC = input_IC.begin(); input_it_IC != input_IC.end(); ++input_it_IC) {
	input_it_IC.coords(pt_coords_IC, mfa_data.min_dim, mfa_data.min_dim);
	input_it_IC.coords(pt_coords_IC2, mfa_data2.min_dim, mfa_data2.min_dim);
	input_it_IC.coords(pt_coords_IC3, mfa_data3.min_dim, mfa_data3.min_dim);
	Q_IC.row(input_it_IC.idx()) = pt_coords_IC;
	Q_IC2.row(input_it_IC.idx()) = pt_coords_IC2;
	Q_IC3.row(input_it_IC.idx()) = pt_coords_IC3;
      }
      // Initialize Q_BC
      Q_BCx.resize(bcline_x); Q_BCx2.resize(bcline_x); Q_BCx3.resize(bcline_x);
      VectorX<T> pt_coords_BC(mfa_data.dim()), pt_coords_BC2(mfa_data2.dim()), pt_coords_BC3(mfa_data3.dim());
      for (auto it = input_BCx.begin(), end = input_BCx.end(); it != end; ++it) {
	it.coords(pt_coords_BC, mfa_data.min_dim, mfa_data.min_dim);
	it.coords(pt_coords_BC2, mfa_data2.min_dim, mfa_data2.min_dim);
	it.coords(pt_coords_BC3, mfa_data3.min_dim, mfa_data3.min_dim);
	Q_BCx.row(it.idx()) = pt_coords_BC; Q_BCx2.row(it.idx()) = pt_coords_BC2; Q_BCx3.row(it.idx()) = pt_coords_BC3;
      }

      Q_BCy.resize(bcline_y); Q_BCy2.resize(bcline_y); Q_BCy3.resize(bcline_y);
      VectorX<T> pt_coords_BCy(mfa_data.dim()), pt_coords_BCy2(mfa_data2.dim()), pt_coords_BCy3(mfa_data3.dim());
      for (auto it = input_BCy.begin(), end = input_BCy.end(); it != end; ++it) {
	it.coords(pt_coords_BCy, mfa_data.min_dim, mfa_data.min_dim);
	it.coords(pt_coords_BCy2, mfa_data2.min_dim, mfa_data2.min_dim);
	it.coords(pt_coords_BCy3, mfa_data3.min_dim, mfa_data3.min_dim);
	Q_BCy.row(it.idx()) = pt_coords_BCy; Q_BCy2.row(it.idx()) = pt_coords_BCy2; Q_BCy3.row(it.idx()) = pt_coords_BCy3;
      }

      // Initialize Q
     Q.resize(N1t.cols());
     Q2.resize(N2t.cols());
     Q3.resize(N3t.cols());
      VectorX<T> pt_coords(mfa_data.dim());
      VectorX<T> pt_coords2(mfa_data2.dim());
      VectorX<T> pt_coords3(mfa_data3.dim());
      for (auto input_it = input.begin(); input_it != input.end(); ++input_it) {
	input_it.coords(pt_coords, mfa_data.min_dim, mfa_data.min_dim);
	input_it.coords(pt_coords2, mfa_data2.min_dim, mfa_data2.min_dim);
	input_it.coords(pt_coords3, mfa_data3.min_dim, mfa_data3.min_dim);
	Q.row(input_it.idx()) = pt_coords;
	Q2.row(input_it.idx()) = pt_coords2;
	Q3.row(input_it.idx()) = pt_coords3;
      }

      // Assemble one discrete spatial-average row for each represented time.
      // All counts are inferred from the PointSet; no 21x21x1000/1001
      // assumptions enter the objective.
      std::map<T, std::vector<int> > time_groups;
      for (int i = 0; i < input.npts; ++i)
        time_groups[input.domain(i, 2)].push_back(i);
      if (time_groups.empty())
        throw std::invalid_argument("Cannot construct pressure gauge from an empty PointSet");
      if (time_groups.size() == static_cast<size_t>(input.npts))
        throw std::invalid_argument(
            "Per-time pressure gauge requires repeated spatial samples at each time");

      typedef Eigen::Triplet<T> GaugeTriplet;
      std::vector<GaugeTriplet> gauge_triplets;
      gauge_triplets.reserve(input.npts);
      int gauge_row = 0;
      for (typename std::map<T, std::vector<int> >::const_iterator group = time_groups.begin();
           group != time_groups.end(); ++group, ++gauge_row)
      {
        const T weight = T(1) / static_cast<T>(group->second.size());
        for (size_t j = 0; j < group->second.size(); ++j)
          gauge_triplets.push_back(GaugeTriplet(gauge_row, group->second[j], weight));
      }
      G.resize(static_cast<int>(time_groups.size()), input.npts);
      G.setFromTriplets(gauge_triplets.begin(), gauge_triplets.end());
      G.makeCompressed();
      Gt = G.transpose();
      Gt.makeCompressed();

      // Fit and validation rows are supplied by the shared, deterministic
      // reviewer design.  Validation rows are never active in the objective.
      loadRowMask(
        "MFA_DATA_ACTIVE_INDEX_FILE", data_active, data_active_count,
        "data-fit", true);
      loadRowMask(
        "MFA_PDE_ACTIVE_INDEX_FILE", pde_active, pde_active_count,
        "PDE-fit", true);
      loadRowMask(
        "MFA_DATA_VALID_INDEX_FILE", data_validation, data_validation_count,
        "data-validation", false);
      loadRowMask(
        "MFA_PDE_VALID_INDEX_FILE", pde_validation, pde_validation_count,
        "PDE-validation", false);
      initializePhysicsAndIntegralOperators();
    }


  T operator()(VectorType const& P, VectorType& grad)
  {
      if (prefiltered_pde_benchmark)
        return evaluatePrefilteredPde(P, grad);

      static int counter = 0;
      int num_print =1;
      // Check if the matrix multiplication is valid
      if (N1.cols()*3 != P.size()) {
	std::cerr << "Dimension mismatch: N1.cols()*3 = " << N1.cols()*3
		  << ", P.size() = " << P.size() << std::endl;
	return std::numeric_limits<T>::quiet_NaN(); // Return an invalid number if dimensions don't match
      }
      int n = P.size()/3;
      Eigen::VectorXd Px(n);
      Eigen::VectorXd Py(n);
      Eigen::VectorXd Pz(n);
      Px=P.segment(0,n);
      Py=P.segment(n,n); //start_index, length
      Pz=P.segment(2*n,n); //start_index, length


      Eigen::VectorXd u    = N1    * Px;
      Eigen::VectorXd ut   = N1_t  * Px;
      Eigen::VectorXd ux   = N1_x  * Px;
      Eigen::VectorXd uy   = N1_y  * Px;
      Eigen::VectorXd ulap = N1_xx * Px + N1_yy * Px;

      Eigen::VectorXd v    = N1    * Py;
      Eigen::VectorXd vt   = N1_t  * Py;
      Eigen::VectorXd vx   = N1_x  * Py;
      Eigen::VectorXd vy   = N1_y  * Py;
      Eigen::VectorXd vlap = N1_xx * Py + N1_yy * Py;

      // Pressure field values and derivatives
      Eigen::VectorXd p    = N1    * Pz;
      Eigen::VectorXd px   = N1_x  * Pz;
      Eigen::VectorXd py   = N1_y  * Pz;

      // One zero-spatial-mean constraint for every represented time slice.
      // This removes the pressure nullspace p -> p + c(t) without shrinking
      // physical pressure fluctuations.
      Eigen::VectorXd gauge_mean = G * p;

      // Residuals
      Eigen::VectorXd ru = ut + u.cwiseProduct(ux) + v.cwiseProduct(uy) + px - nu * ulap;
      Eigen::VectorXd rv = vt + u.cwiseProduct(vx) + v.cwiseProduct(vy) + py - nu * vlap;
      Eigen::VectorXd rc = ux + vy;

// ---------------------------------------------------------
      // 1. Normalization Constants (True Mean Squared Error)
      // ---------------------------------------------------------
      double num_data = data_active_count; // Independent data-fit rows
      double num_pde = pde_active_count;   // Independent PDE-fit rows
      double num_ic  = input_IC.npts;  // Total IC points
      double num_bcx = input_BCx.npts; // Total X-Boundary points
      double num_bcy = input_BCy.npts; // Total Y-Boundary points
      double num_bcx_active = active_bcx_count;
      double num_bcy_active = active_bcy_count;
      double num_gauge = G.rows();     // One spatial mean per time slice
      Eigen::VectorXd ru_grad = ru.cwiseProduct(pde_active);
      Eigen::VectorXd rv_grad = rv.cwiseProduct(pde_active);
      Eigen::VectorXd rc_grad = rc.cwiseProduct(pde_active);

      // ---------------------------------------------------------
      // Jacobian-transpose products for the momentum blocks.
      // ---------------------------------------------------------

      // Derivatives of masked x-momentum (ru_grad) w.r.t Px, Py, Pz
      Eigen::VectorXd JT_ru_Px = N1t_t * ru_grad
                               + N1t   * (ux.cwiseProduct(ru_grad))
                               + N1t_x * (u.cwiseProduct(ru_grad))
                               + N1t_y * (v.cwiseProduct(ru_grad))
                               - nu * (N1t_xx + N1t_yy) * ru_grad;

      Eigen::VectorXd JT_ru_Py = N1t   * (uy.cwiseProduct(ru_grad)); // Cross-term advection

      Eigen::VectorXd JT_ru_Pz = N1t_x * ru_grad; // Pressure gradient

      // Derivatives of masked y-momentum (rv_grad) w.r.t Px, Py, Pz
      Eigen::VectorXd JT_rv_Px = N1t   * (vx.cwiseProduct(rv_grad)); // Cross-term advection

      Eigen::VectorXd JT_rv_Py = N1t_t * rv_grad
                               + N1t_x * (u.cwiseProduct(rv_grad))
                               + N1t   * (vy.cwiseProduct(rv_grad))
                               + N1t_y * (v.cwiseProduct(rv_grad))
                               - nu * (N1t_xx + N1t_yy) * rv_grad;

      Eigen::VectorXd JT_rv_Pz = N1t_y * rv_grad; // Pressure gradient

      // ---------------------------------------------------------
      // 3. Divergence Jacobian transpose contribution (Masked)
      // ---------------------------------------------------------
      Eigen::VectorXd JT_div_Px = N1t_x * rc_grad;
      Eigen::VectorXd JT_div_Py = N1t_y * rc_grad;
      Eigen::VectorXd data_u_raw = u - Q;
      Eigen::VectorXd data_v_raw = v - Q2;
      Eigen::VectorXd data_p_raw = p - Q3;
      Eigen::VectorXd data_u = data_u_raw.cwiseProduct(data_active);
      Eigen::VectorXd data_v = data_v_raw.cwiseProduct(data_active);
      Eigen::VectorXd data_p = data_p_raw.cwiseProduct(data_active);


      // ---------------------------------------------------------
      // 2. Exact Gradients (Perfectly Normalized to MSE)
      // ---------------------------------------------------------
      Eigen::VectorXd gradPx;
      Eigen::VectorXd gradPy;
      if (!physics_time_mask_active)
      {
        // Historical branch kept verbatim for the zero-weight/default-time
        // regression.
        gradPx =
          (2.0 / num_data) * lambda_data_u * (N1t * data_u)
        + (2.0 / num_pde) * (lambda_xmom * JT_ru_Px + lambda_ymom * JT_rv_Px + lambda_div * JT_div_Px)
        + (2.0 / num_ic)  * lambda_ic_u * (ICt * (IC * Px - Q_IC))
        + lambda_bc_u * ( (2.0 / num_bcx) * (Bxt * (Bx * Px - Q_BCx)) + (2.0 / num_bcy) * (Byt * (By * Px - Q_BCy)) );

        gradPy =
          (2.0 / num_data) * lambda_data_v * (N1t * data_v)
        + (2.0 / num_pde) * (lambda_xmom * JT_ru_Py + lambda_ymom * JT_rv_Py + lambda_div * JT_div_Py)
        + (2.0 / num_ic)  * lambda_ic_v * (ICt * (IC * Py - Q_IC2))
        + lambda_bc_v * ( (2.0 / num_bcx) * (Bxt * (Bx * Py - Q_BCx2)) + (2.0 / num_bcy) * (Byt * (By * Py - Q_BCy2)) );
      }
      else
      {
        const Eigen::VectorXd bcx_u =
          (Bx * Px - Q_BCx).cwiseProduct(bcx_physics_mask);
        const Eigen::VectorXd bcy_u =
          (By * Px - Q_BCy).cwiseProduct(bcy_physics_mask);
        const Eigen::VectorXd bcx_v =
          (Bx * Py - Q_BCx2).cwiseProduct(bcx_physics_mask);
        const Eigen::VectorXd bcy_v =
          (By * Py - Q_BCy2).cwiseProduct(bcy_physics_mask);

        gradPx =
          (2.0 / num_data) * lambda_data_u * (N1t * data_u)
        + (2.0 / num_pde) * (lambda_xmom * JT_ru_Px + lambda_ymom * JT_rv_Px + lambda_div * JT_div_Px)
        + (2.0 / num_ic)  * lambda_ic_u * (ICt * (IC * Px - Q_IC))
        + lambda_bc_u * (
            (2.0 / num_bcx_active) * (Bxt * bcx_u) +
            (2.0 / num_bcy_active) * (Byt * bcy_u));

        gradPy =
          (2.0 / num_data) * lambda_data_v * (N1t * data_v)
        + (2.0 / num_pde) * (lambda_xmom * JT_ru_Py + lambda_ymom * JT_rv_Py + lambda_div * JT_div_Py)
        + (2.0 / num_ic)  * lambda_ic_v * (ICt * (IC * Py - Q_IC2))
        + lambda_bc_v * (
            (2.0 / num_bcx_active) * (Bxt * bcx_v) +
            (2.0 / num_bcy_active) * (Byt * bcy_v));
      }

      Eigen::VectorXd grad_gauge_Pz =
        (2.0 * lambda_gauge / num_gauge) * (N1t * (Gt * gauge_mean));

      Eigen::VectorXd gradPz;
      if (!physics_time_mask_active)
      {
        gradPz =
          (2.0 / num_data) * lambda_data_p * (N1t * data_p)
        + (2.0 / num_pde) * (lambda_xmom * JT_ru_Pz + lambda_ymom * JT_rv_Pz)
        + (2.0 / num_ic)  * lambda_ic_p * (ICt * (IC * Pz - Q_IC3))
        + lambda_bc_p * ( (2.0 / num_bcx) * (Bxt * (Bx * Pz - Q_BCx3)) + (2.0 / num_bcy) * (Byt * (By * Pz - Q_BCy3)) )
        + grad_gauge_Pz;
      }
      else
      {
        const Eigen::VectorXd bcx_p =
          (Bx * Pz - Q_BCx3).cwiseProduct(bcx_physics_mask);
        const Eigen::VectorXd bcy_p =
          (By * Pz - Q_BCy3).cwiseProduct(bcy_physics_mask);
        gradPz =
          (2.0 / num_data) * lambda_data_p * (N1t * data_p)
        + (2.0 / num_pde) * (lambda_xmom * JT_ru_Pz + lambda_ymom * JT_rv_Pz)
        + (2.0 / num_ic)  * lambda_ic_p * (ICt * (IC * Pz - Q_IC3))
        + lambda_bc_p * (
            (2.0 / num_bcx_active) * (Bxt * bcx_p) +
            (2.0 / num_bcy_active) * (Byt * bcy_p))
        + grad_gauge_Pz;
      }

      NSIntegralBalanceEvaluation<T> integral_fit;
      NSKnotSpanGLBalanceEvaluation<T> knot_span_integral_fit;
      if (integral_operators)
      {
        const bool integral_objective_enabled = integral_config.enabled();
        integral_fit = NSIntegralBalance<T>::evaluate(
          integral_operators->fit,
          u, ut, ux, uy, v, vt, vx, vy, p,
          static_cast<T>(nu),
          integral_config.lambda_mass,
          integral_config.lambda_xmom,
          integral_config.lambda_ymom,
          integral_objective_enabled);
        if (integral_objective_enabled)
        {
          gradPx +=
            N1t * integral_fit.adjoints.u +
            N1t_t * integral_fit.adjoints.ut +
            N1t_x * integral_fit.adjoints.ux +
            N1t_y * integral_fit.adjoints.uy;
          gradPy +=
            N1t * integral_fit.adjoints.v +
            N1t_t * integral_fit.adjoints.vt +
            N1t_x * integral_fit.adjoints.vx +
            N1t_y * integral_fit.adjoints.vy;
          gradPz += N1t * integral_fit.adjoints.p;
        }
      }
      else if (knot_span_integral_operators)
      {
        const bool integral_objective_enabled = integral_config.enabled();
        knot_span_integral_fit = NSKnotSpanGLBalance<T>::evaluate(
          knot_span_integral_operators->fit,
          Px, Py, Pz,
          static_cast<T>(nu),
          integral_config.lambda_mass,
          integral_config.lambda_xmom,
          integral_config.lambda_ymom,
          integral_objective_enabled);
        if (integral_objective_enabled)
        {
          gradPx += knot_span_integral_fit.coefficient_adjoints.u;
          gradPy += knot_span_integral_fit.coefficient_adjoints.v;
          gradPz += knot_span_integral_fit.coefficient_adjoints.p;
        }
      }

      grad.segment(0,n)   = gradPx;
      grad.segment(n,n)   = gradPy;
      grad.segment(2*n,n) = gradPz;
      double loss_data = (lambda_data_u * data_u.squaredNorm() + lambda_data_v * data_v.squaredNorm() + lambda_data_p * data_p.squaredNorm()) / num_data;
      double loss_mom  = (lambda_xmom * ru_grad.squaredNorm() + lambda_ymom * rv_grad.squaredNorm()) / num_pde;
      double loss_div  = (lambda_div  * rc_grad.squaredNorm()) / num_pde;

      double loss_ic   = (lambda_ic_u * (IC * Px - Q_IC).squaredNorm() +
                          lambda_ic_v * (IC * Py - Q_IC2).squaredNorm() +
                          lambda_ic_p * (IC * Pz - Q_IC3).squaredNorm()) / num_ic;

      double loss_bcx;
      double loss_bcy;
      if (!physics_time_mask_active)
      {
        // Historical branch kept verbatim for regression.
        loss_bcx  = lambda_bc_u * (Bx * Px - Q_BCx).squaredNorm() / num_bcx +
                     lambda_bc_v * (Bx * Py - Q_BCx2).squaredNorm() / num_bcx +
                     lambda_bc_p * (Bx * Pz - Q_BCx3).squaredNorm() / num_bcx;

        loss_bcy  = lambda_bc_u * (By * Px - Q_BCy).squaredNorm() / num_bcy +
                     lambda_bc_v * (By * Py - Q_BCy2).squaredNorm() / num_bcy +
                     lambda_bc_p * (By * Pz - Q_BCy3).squaredNorm() / num_bcy;
      }
      else
      {
        loss_bcx =
          lambda_bc_u *
            (Bx * Px - Q_BCx).cwiseProduct(
              bcx_physics_mask).squaredNorm() / num_bcx_active +
          lambda_bc_v *
            (Bx * Py - Q_BCx2).cwiseProduct(
              bcx_physics_mask).squaredNorm() / num_bcx_active +
          lambda_bc_p *
            (Bx * Pz - Q_BCx3).cwiseProduct(
              bcx_physics_mask).squaredNorm() / num_bcx_active;

        loss_bcy =
          lambda_bc_u *
            (By * Px - Q_BCy).cwiseProduct(
              bcy_physics_mask).squaredNorm() / num_bcy_active +
          lambda_bc_v *
            (By * Py - Q_BCy2).cwiseProduct(
              bcy_physics_mask).squaredNorm() / num_bcy_active +
          lambda_bc_p *
            (By * Pz - Q_BCy3).cwiseProduct(
              bcy_physics_mask).squaredNorm() / num_bcy_active;
      }

      double loss_bc   = loss_bcx + loss_bcy;
      double loss_gauge = lambda_gauge * gauge_mean.squaredNorm() / num_gauge;
      double loss = loss_data + loss_mom + loss_div + loss_ic + loss_bc + loss_gauge;
      if (integral_config.enabled())
        loss += integral_config.quadrature == "gl4"
          ? knot_span_integral_fit.weighted_loss
          : integral_fit.weighted_loss;
            if (!suppress_diagnostics && counter % num_print == 0)
	{
          // These are the exact unweighted MSE blocks used by the objective.
          // The first logged value below is the actual weighted scalar objective.
          double scaled_data_u = data_u.squaredNorm() / num_data;
          double scaled_data_v = data_v.squaredNorm() / num_data;
          double scaled_data_p = data_p.squaredNorm() / num_data;
          double scaled_mom_u  = ru_grad.squaredNorm() / num_pde;
          double scaled_mom_v  = rv_grad.squaredNorm() / num_pde;
          double scaled_div    = rc_grad.squaredNorm() / num_pde;
          double scaled_ic_u   = (IC * Px - Q_IC).squaredNorm() / num_ic;
          double scaled_ic_v   = (IC * Py - Q_IC2).squaredNorm() / num_ic;
          double scaled_ic_p   = (IC * Pz - Q_IC3).squaredNorm() / num_ic;
          double scaled_bc_u;
          double scaled_bc_v;
          double scaled_bc_p;
          if (!physics_time_mask_active)
          {
            scaled_bc_u = (Bx * Px - Q_BCx).squaredNorm() / num_bcx
                        + (By * Px - Q_BCy).squaredNorm() / num_bcy;
            scaled_bc_v = (Bx * Py - Q_BCx2).squaredNorm() / num_bcx
                        + (By * Py - Q_BCy2).squaredNorm() / num_bcy;
            scaled_bc_p = (Bx * Pz - Q_BCx3).squaredNorm() / num_bcx
                        + (By * Pz - Q_BCy3).squaredNorm() / num_bcy;
          }
          else
          {
            scaled_bc_u =
              (Bx * Px - Q_BCx).cwiseProduct(
                bcx_physics_mask).squaredNorm() / num_bcx_active +
              (By * Px - Q_BCy).cwiseProduct(
                bcy_physics_mask).squaredNorm() / num_bcy_active;
            scaled_bc_v =
              (Bx * Py - Q_BCx2).cwiseProduct(
                bcx_physics_mask).squaredNorm() / num_bcx_active +
              (By * Py - Q_BCy2).cwiseProduct(
                bcy_physics_mask).squaredNorm() / num_bcy_active;
            scaled_bc_p =
              (Bx * Pz - Q_BCx3).cwiseProduct(
                bcx_physics_mask).squaredNorm() / num_bcx_active +
              (By * Pz - Q_BCy3).cwiseProduct(
                bcy_physics_mask).squaredNorm() / num_bcy_active;
          }
	  double scaled_gauge  = gauge_mean.squaredNorm() / num_gauge;
          const double unavailable = std::numeric_limits<double>::quiet_NaN();
          double valid_data_u = data_validation_count > T(0)
            ? data_u_raw.cwiseProduct(data_validation).squaredNorm() / data_validation_count
            : unavailable;
          double valid_data_v = data_validation_count > T(0)
            ? data_v_raw.cwiseProduct(data_validation).squaredNorm() / data_validation_count
            : unavailable;
          double valid_data_p = data_validation_count > T(0)
            ? data_p_raw.cwiseProduct(data_validation).squaredNorm() / data_validation_count
            : unavailable;
          double valid_mom_u = pde_validation_count > T(0)
            ? ru.cwiseProduct(pde_validation).squaredNorm() / pde_validation_count
            : unavailable;
          double valid_mom_v = pde_validation_count > T(0)
            ? rv.cwiseProduct(pde_validation).squaredNorm() / pde_validation_count
            : unavailable;
          double valid_div = pde_validation_count > T(0)
            ? rc.cwiseProduct(pde_validation).squaredNorm() / pde_validation_count
            : unavailable;

          NSIntegralBalanceEvaluation<T> integral_validation;
          if (integral_operators)
            integral_validation = NSIntegralBalance<T>::evaluate(
              integral_operators->validation,
              u, ut, ux, uy, v, vt, vx, vy, p,
              static_cast<T>(nu),
              T(0), T(0), T(0), false);
          NSIntegralBalanceEvaluation<T> global_integral_validation;
          if (global_integral_operators)
            global_integral_validation = NSIntegralBalance<T>::evaluate(
              global_integral_operators->validation,
              u, ut, ux, uy, v, vt, vx, vy, p,
              static_cast<T>(nu),
              T(0), T(0), T(0), false);
          NSKnotSpanGLBalanceEvaluation<T>
            knot_span_integral_validation;
          if (knot_span_integral_operators)
            knot_span_integral_validation =
              NSKnotSpanGLBalance<T>::evaluate(
                knot_span_integral_operators->validation,
                Px, Py, Pz,
                static_cast<T>(nu),
                T(0), T(0), T(0), false);
          NSKnotSpanGLBalanceEvaluation<T>
            global_knot_span_integral_validation;
          if (global_knot_span_integral_operators)
            global_knot_span_integral_validation =
              NSKnotSpanGLBalance<T>::evaluate(
                global_knot_span_integral_operators->validation,
                Px, Py, Pz,
                static_cast<T>(nu),
                T(0), T(0), T(0), false);

          const bool knot_span_gl4 =
            static_cast<bool>(knot_span_integral_operators);
          const T fit_integral_mass = knot_span_gl4
            ? knot_span_integral_fit.mse_mass : integral_fit.mse_mass;
          const T fit_integral_xmom = knot_span_gl4
            ? knot_span_integral_fit.mse_xmom : integral_fit.mse_xmom;
          const T fit_integral_ymom = knot_span_gl4
            ? knot_span_integral_fit.mse_ymom : integral_fit.mse_ymom;
          const T validation_integral_mass = knot_span_gl4
            ? knot_span_integral_validation.mse_mass
            : integral_validation.mse_mass;
          const T validation_integral_xmom = knot_span_gl4
            ? knot_span_integral_validation.mse_xmom
            : integral_validation.mse_xmom;
          const T validation_integral_ymom = knot_span_gl4
            ? knot_span_integral_validation.mse_ymom
            : integral_validation.mse_ymom;
          const T global_validation_integral_mass = knot_span_gl4
            ? global_knot_span_integral_validation.mse_mass
            : global_integral_validation.mse_mass;
          const T global_validation_integral_xmom = knot_span_gl4
            ? global_knot_span_integral_validation.mse_xmom
            : global_integral_validation.mse_xmom;
          const T global_validation_integral_ymom = knot_span_gl4
            ? global_knot_span_integral_validation.mse_ymom
            : global_integral_validation.mse_ymom;
          const int integral_fit_rows = knot_span_gl4
            ? knot_span_integral_operators->fit.rows()
            : (integral_operators ? integral_operators->fit.rows() : 0);
          const int integral_validation_rows = knot_span_gl4
            ? knot_span_integral_operators->validation.rows()
            : (integral_operators ? integral_operators->validation.rows() : 0);
          const int global_integral_validation_rows = knot_span_gl4
            ? global_knot_span_integral_operators->validation.rows()
            : (global_integral_operators
                ? global_integral_operators->validation.rows() : 0);

	  std::ofstream loss_file("loss_BFGS_navier_stokes.txt", std::ios::out | std::ios::app);
	  if (loss_file.is_open())
	    {
	      loss_file
		<< counter << " "
		<< loss << " "
		<< scaled_data_u << " "
		<< scaled_data_v << " "
		<< scaled_data_p << " "
		<< scaled_mom_u  << " "
		<< scaled_mom_v  << " "
		<< scaled_div    << " "
		<< scaled_ic_u   << " "
		<< scaled_ic_v   << " "
		<< scaled_ic_p   << " "
		<< scaled_bc_u   << " "
		<< scaled_bc_v   << " "
		<< scaled_bc_p   << " "
		<< scaled_gauge  << " "    // <--- Added gauge
                << valid_data_u << " "
                << valid_data_v << " "
                << valid_data_p << " "
                << valid_mom_u << " "
                << valid_mom_v << " "
                << valid_div << " "
		<< std::endl;

	      loss_file.close();
	    }
	  else
	    {
	      std::cerr << "Failed to open file: loss_BFGS_navier_stokes.txt" << std::endl;
	    }

          if (physics_time_mask_active || integral_config.operatorsRequired())
          {
            std::ios_base::openmode mode =
              std::ios::out |
              (counter == 0 ? std::ios::trunc : std::ios::app);
            std::ofstream hybrid_file(
              "loss_BFGS_navier_stokes_hybrid.tsv", mode);
            if (!hybrid_file)
              throw std::runtime_error(
                "Cannot open loss_BFGS_navier_stokes_hybrid.tsv");
            if (counter == 0)
              hybrid_file
                << "evaluation\tobjective\tfit_data_u\tfit_data_v\tfit_data_p"
                << "\tfit_strong_xmom\tfit_strong_ymom\tfit_strong_div"
                << "\tfit_ic_u\tfit_ic_v\tfit_ic_p"
                << "\tfit_bc_u\tfit_bc_v\tfit_bc_p\tfit_gauge"
                << "\tval_data_u\tval_data_v\tval_data_p"
                << "\tval_strong_xmom\tval_strong_ymom\tval_strong_div"
                << "\tfit_int_mass\tfit_int_xmom\tfit_int_ymom"
                << "\tval_int_mass\tval_int_xmom\tval_int_ymom"
                << "\tval_global_mass\tval_global_xmom\tval_global_ymom"
                << "\tactive_bc_x\tactive_bc_y"
                << "\tint_fit_rows\tint_validation_rows"
                << "\tglobal_int_validation_rows\n";
            hybrid_file
              << std::setprecision(17)
              << counter << "\t" << loss
              << "\t" << scaled_data_u
              << "\t" << scaled_data_v
              << "\t" << scaled_data_p
              << "\t" << scaled_mom_u
              << "\t" << scaled_mom_v
              << "\t" << scaled_div
              << "\t" << scaled_ic_u
              << "\t" << scaled_ic_v
              << "\t" << scaled_ic_p
              << "\t" << scaled_bc_u
              << "\t" << scaled_bc_v
              << "\t" << scaled_bc_p
              << "\t" << scaled_gauge
              << "\t" << valid_data_u
              << "\t" << valid_data_v
              << "\t" << valid_data_p
              << "\t" << valid_mom_u
              << "\t" << valid_mom_v
              << "\t" << valid_div
              << "\t" << fit_integral_mass
              << "\t" << fit_integral_xmom
              << "\t" << fit_integral_ymom
              << "\t" << validation_integral_mass
              << "\t" << validation_integral_xmom
              << "\t" << validation_integral_ymom
              << "\t" << global_validation_integral_mass
              << "\t" << global_validation_integral_xmom
              << "\t" << global_validation_integral_ymom
              << "\t" << active_bcx_count
              << "\t" << active_bcy_count
              << "\t" << integral_fit_rows
              << "\t" << integral_validation_rows
              << "\t" << global_integral_validation_rows
              << "\n";
          }
	}
      if (!suppress_diagnostics)
        ++counter;
      return loss;

}

    T evaluatePrefilteredPde(const VectorType& P, VectorType& grad)
    {
      if (!prefiltered_pde_benchmark)
        throw std::logic_error(
          "evaluatePrefilteredPde requires MFA_PREFILTERED_PDE_BENCHMARK=1");
      if (N1.cols() * 3 != P.size())
        throw std::invalid_argument(
          "NS prefiltered benchmark coefficient-size mismatch");

      const int coefficient_count = P.size() / 3;
      const Eigen::VectorXd pu = P.segment(0, coefficient_count);
      const Eigen::VectorXd pv = P.segment(
        coefficient_count, coefficient_count);
      const Eigen::VectorXd pp = P.segment(
        2 * coefficient_count, coefficient_count);

      const Eigen::VectorXd u = N1 * pu;
      const Eigen::VectorXd ut = N1_t * pu;
      const Eigen::VectorXd ux = N1_x * pu;
      const Eigen::VectorXd uy = N1_y * pu;
      const Eigen::VectorXd ulap = N1_xx * pu + N1_yy * pu;

      const Eigen::VectorXd v = N1 * pv;
      const Eigen::VectorXd vt = N1_t * pv;
      const Eigen::VectorXd vx = N1_x * pv;
      const Eigen::VectorXd vy = N1_y * pv;
      const Eigen::VectorXd vlap = N1_xx * pv + N1_yy * pv;

      const Eigen::VectorXd px = N1_x * pp;
      const Eigen::VectorXd py = N1_y * pp;
      const Eigen::VectorXd ru =
        ut + u.cwiseProduct(ux) + v.cwiseProduct(uy) + px - nu * ulap;
      const Eigen::VectorXd rv =
        vt + u.cwiseProduct(vx) + v.cwiseProduct(vy) + py - nu * vlap;
      const Eigen::VectorXd rc = ux + vy;

      const Eigen::VectorXd jt_ru_u =
        N1t_t * ru
        + N1t * (ux.cwiseProduct(ru))
        + N1t_x * (u.cwiseProduct(ru))
        + N1t_y * (v.cwiseProduct(ru))
        - nu * (N1t_xx + N1t_yy) * ru;
      const Eigen::VectorXd jt_ru_v =
        N1t * (uy.cwiseProduct(ru));
      const Eigen::VectorXd jt_ru_p = N1t_x * ru;

      const Eigen::VectorXd jt_rv_u =
        N1t * (vx.cwiseProduct(rv));
      const Eigen::VectorXd jt_rv_v =
        N1t_t * rv
        + N1t_x * (u.cwiseProduct(rv))
        + N1t * (vy.cwiseProduct(rv))
        + N1t_y * (v.cwiseProduct(rv))
        - nu * (N1t_xx + N1t_yy) * rv;
      const Eigen::VectorXd jt_rv_p = N1t_y * rv;
      const Eigen::VectorXd jt_div_u = N1t_x * rc;
      const Eigen::VectorXd jt_div_v = N1t_y * rc;

      const T inverse_rows =
        T(1) / static_cast<T>(prefiltered_pde_rows.size());
      grad.segment(0, coefficient_count) =
        T(2) * inverse_rows * (jt_ru_u + jt_rv_u + jt_div_u);
      grad.segment(coefficient_count, coefficient_count) =
        T(2) * inverse_rows * (jt_ru_v + jt_rv_v + jt_div_v);
      grad.segment(2 * coefficient_count, coefficient_count) =
        T(2) * inverse_rows * (jt_ru_p + jt_rv_p);
      return (ru.squaredNorm() + rv.squaredNorm() + rc.squaredNorm())
        * inverse_rows;
    }

    size_t prefilteredPdeRows() const
    {
      return prefiltered_pde_rows.size();
    }

    double prefilteredConstructionSeconds() const
    {
      return prefiltered_construction_seconds;
    }

    bool checkDirectionalGradients(
        const VectorType& P,
        int               checks,
        T                 relative_step,
        T                 tolerance)
    {
      if (checks <= 0)
        return true;

      suppress_diagnostics = true;
      VectorType analytical_gradient(P.size());
      const T f0 = (*this)(P, analytical_gradient);
      if (!std::isfinite(f0) || !analytical_gradient.allFinite())
      {
        suppress_diagnostics = false;
        std::cerr << "NS gradient check failed: non-finite objective or gradient\n";
        return false;
      }

      std::mt19937_64 generator(20260723u);
      std::normal_distribution<T> normal(T(0), T(1));
      const bool multi_epsilon =
        nsIntegralStrictBoolean("MFA_GRADIENT_CHECK_MULTI_EPS", false);
      std::vector<T> relative_steps;
      if (multi_epsilon)
      {
        relative_steps.push_back(T(1.0e-2));
        relative_steps.push_back(T(1.0e-3));
        relative_steps.push_back(T(1.0e-4));
        relative_steps.push_back(T(1.0e-5));
        relative_steps.push_back(T(1.0e-6));
        relative_steps.push_back(T(1.0e-7));
      }
      else
        relative_steps.push_back(relative_step);
      bool passed = true;

      for (int check = 0; check < checks; ++check)
      {
        VectorType direction(P.size());
        for (int i = 0; i < direction.size(); ++i)
          direction(i) = normal(generator);
        direction.normalize();

        const T analytical = analytical_gradient.dot(direction);
        T minimum_relative_error = std::numeric_limits<T>::infinity();
        std::vector<T> taylor_remainders;
        for (size_t step_index = 0;
             step_index < relative_steps.size(); ++step_index)
        {
          const T step =
            relative_steps[step_index] * std::max<T>(T(1), P.norm());
          VectorType throwaway(P.size());
          const T f_plus  = (*this)(P + step * direction, throwaway);
          const T f_minus = (*this)(P - step * direction, throwaway);
          const T finite_difference =
            (f_plus - f_minus) / (T(2) * step);
          const T denominator =
            std::max<T>(
              T(1),
              std::max<T>(
                std::abs(finite_difference), std::abs(analytical)));
          const T relative_error =
            std::abs(finite_difference - analytical) / denominator;
          const T taylor_remainder =
            std::abs(f_plus - f0 - step * analytical);
          minimum_relative_error =
            std::min(minimum_relative_error, relative_error);
          taylor_remainders.push_back(taylor_remainder);

          std::cout << std::setprecision(16)
                    << "NS directional gradient check "
                    << (check + 1) << "/" << checks
                    << " relative_step=" << relative_steps[step_index]
                    << " analytic=" << analytical
                    << " finite_difference=" << finite_difference
                    << " relative_error=" << relative_error
                    << " taylor_remainder=" << taylor_remainder
                    << "\n";
        }

        bool second_order_seen = !multi_epsilon;
        if (multi_epsilon)
        {
          for (size_t i = 1; i < taylor_remainders.size(); ++i)
          {
            if (taylor_remainders[i] > T(100) *
                std::numeric_limits<T>::epsilon() *
                std::max<T>(T(1), std::abs(f0)))
            {
              const T slope =
                std::log(taylor_remainders[i - 1] /
                         taylor_remainders[i]) /
                std::log(relative_steps[i - 1] / relative_steps[i]);
              std::cout << std::setprecision(16)
                        << "NS Taylor slope check=" << (check + 1)
                        << " interval=" << (i - 1) << "-" << i
                        << " slope=" << slope << "\n";
              if (std::isfinite(slope) && slope >= T(1.7) &&
                  slope <= T(2.3))
                second_order_seen = true;
            }
          }
        }
        if (!std::isfinite(minimum_relative_error) ||
            minimum_relative_error > tolerance ||
            !second_order_seen)
          passed = false;
      }

      suppress_diagnostics = false;
      return passed;
    }

    // Evaluate an exact, no-optimization decomposition of the production
    // objective gradient at one fixed coefficient vector.  Each atomic term
    // is obtained by calling this same operator with every weight zero except
    // the selected unit weight.  This deliberately avoids maintaining a
    // second copy of the nonlinear residual or adjoint algebra.
    void writeObjectiveGradientAudit(
        const VectorType& P,
        const std::string& vector_path,
        const std::string& term_path)
    {
      if (prefiltered_pde_benchmark)
        throw std::invalid_argument(
          "Objective-gradient audit is unavailable in prefiltered mode");
      if (P.size() != n || P.size() % 3 != 0)
        throw std::invalid_argument(
          "Objective-gradient audit received an invalid coefficient vector");
      if (vector_path.empty() || term_path.empty() || vector_path == term_path)
        throw std::invalid_argument(
          "Objective-gradient audit requires two distinct output paths");
      if (!knot_span_integral_operators || integral_config.quadrature != "gl4")
        throw std::invalid_argument(
          "Objective-gradient audit requires the declared knot-span GL4 operators");

      struct WeightState
      {
        T data_u, data_v, data_p;
        T strong_xmom, strong_ymom, strong_mass;
        T ic_u, ic_v, ic_p;
        T bc_u, bc_v, bc_p;
        T gauge;
        T gl4_mass, gl4_xmom, gl4_ymom;
      };
      const WeightState original = {
        lambda_data_u, lambda_data_v, lambda_data_p,
        lambda_xmom, lambda_ymom, lambda_div,
        lambda_ic_u, lambda_ic_v, lambda_ic_p,
        lambda_bc_u, lambda_bc_v, lambda_bc_p,
        lambda_gauge,
        integral_config.lambda_mass,
        integral_config.lambda_xmom,
        integral_config.lambda_ymom
      };

      const auto restore_weights = [&]()
      {
        lambda_data_u = original.data_u;
        lambda_data_v = original.data_v;
        lambda_data_p = original.data_p;
        lambda_xmom = original.strong_xmom;
        lambda_ymom = original.strong_ymom;
        lambda_div = original.strong_mass;
        lambda_ic_u = original.ic_u;
        lambda_ic_v = original.ic_v;
        lambda_ic_p = original.ic_p;
        lambda_bc_u = original.bc_u;
        lambda_bc_v = original.bc_v;
        lambda_bc_p = original.bc_p;
        lambda_gauge = original.gauge;
        integral_config.lambda_mass = original.gl4_mass;
        integral_config.lambda_xmom = original.gl4_xmom;
        integral_config.lambda_ymom = original.gl4_ymom;
      };
      const auto zero_weights = [&]()
      {
        lambda_data_u = T(0);
        lambda_data_v = T(0);
        lambda_data_p = T(0);
        lambda_xmom = T(0);
        lambda_ymom = T(0);
        lambda_div = T(0);
        lambda_ic_u = T(0);
        lambda_ic_v = T(0);
        lambda_ic_p = T(0);
        lambda_bc_u = T(0);
        lambda_bc_v = T(0);
        lambda_bc_p = T(0);
        lambda_gauge = T(0);
        integral_config.lambda_mass = T(0);
        integral_config.lambda_xmom = T(0);
        integral_config.lambda_ymom = T(0);
      };

      struct TermRecord
      {
        std::string name;
        T original_weight;
        T unit_objective;
        VectorType unit_gradient;
      };
      const char* names[] = {
        "data_u", "data_v", "data_p",
        "strong_xmom", "strong_ymom", "strong_mass",
        "ic_u", "ic_v", "ic_p",
        "bc_u", "bc_v", "bc_p",
        "gauge", "gl4_mass", "gl4_xmom", "gl4_ymom"
      };
      const T weights[] = {
        original.data_u, original.data_v, original.data_p,
        original.strong_xmom, original.strong_ymom, original.strong_mass,
        original.ic_u, original.ic_v, original.ic_p,
        original.bc_u, original.bc_v, original.bc_p,
        original.gauge, original.gl4_mass,
        original.gl4_xmom, original.gl4_ymom
      };
      const int term_count = static_cast<int>(sizeof(names) / sizeof(names[0]));
      std::vector<TermRecord> records;
      records.reserve(term_count);
      const bool original_suppress_diagnostics = suppress_diagnostics;
      suppress_diagnostics = true;

      VectorType full_gradient_before(P.size());
      VectorType full_gradient(P.size());
      T full_objective_before = T(0);
      T full_objective = T(0);
      try
      {
        restore_weights();
        full_objective_before = (*this)(P, full_gradient_before);
        if (!std::isfinite(full_objective_before) ||
            !full_gradient_before.allFinite())
          throw std::runtime_error(
            "Non-finite pre-decomposition objective-gradient evaluation");
        for (int term = 0; term < term_count; ++term)
        {
          zero_weights();
          switch (term)
          {
            case 0:  lambda_data_u = T(1); break;
            case 1:  lambda_data_v = T(1); break;
            case 2:  lambda_data_p = T(1); break;
            case 3:  lambda_xmom = T(1); break;
            case 4:  lambda_ymom = T(1); break;
            case 5:  lambda_div = T(1); break;
            case 6:  lambda_ic_u = T(1); break;
            case 7:  lambda_ic_v = T(1); break;
            case 8:  lambda_ic_p = T(1); break;
            case 9:  lambda_bc_u = T(1); break;
            case 10: lambda_bc_v = T(1); break;
            case 11: lambda_bc_p = T(1); break;
            case 12: lambda_gauge = T(1); break;
            case 13: integral_config.lambda_mass = T(1); break;
            case 14: integral_config.lambda_xmom = T(1); break;
            case 15: integral_config.lambda_ymom = T(1); break;
            default: throw std::logic_error("Unknown objective-gradient audit term");
          }
          VectorType unit_gradient(P.size());
          const T unit_objective = (*this)(P, unit_gradient);
          if (!std::isfinite(unit_objective) || !unit_gradient.allFinite())
            throw std::runtime_error(
              std::string("Non-finite objective-gradient audit term: ") +
              names[term]);
          TermRecord record;
          record.name = names[term];
          record.original_weight = weights[term];
          record.unit_objective = unit_objective;
          record.unit_gradient.swap(unit_gradient);
          records.push_back(record);
        }
        restore_weights();
        full_objective = (*this)(P, full_gradient);
        if (!std::isfinite(full_objective) || !full_gradient.allFinite())
          throw std::runtime_error(
            "Non-finite direct objective-gradient audit evaluation");
      }
      catch (...)
      {
        restore_weights();
        suppress_diagnostics = original_suppress_diagnostics;
        throw;
      }
      restore_weights();
      suppress_diagnostics = original_suppress_diagnostics;

      const VectorType restoration_residual =
        full_gradient - full_gradient_before;
      const T restoration_gradient_linf =
        restoration_residual.template lpNorm<Eigen::Infinity>();
      const T restoration_objective_abs =
        std::abs(full_objective - full_objective_before);
      if (restoration_gradient_linf > T(1.0e-15) ||
          restoration_objective_abs > T(1.0e-15))
        throw std::runtime_error(
          "Objective-gradient audit did not restore the production weights");

      VectorType weighted_sum = VectorType::Zero(P.size());
      T weighted_objective_sum = T(0);
      for (int term = 0; term < term_count; ++term)
      {
        weighted_sum.noalias() +=
          records[term].original_weight * records[term].unit_gradient;
        weighted_objective_sum +=
          records[term].original_weight * records[term].unit_objective;
      }
      const VectorType reconstruction_residual = full_gradient - weighted_sum;
      const T gradient_relative_l2 =
        reconstruction_residual.norm() /
        std::max<T>(full_gradient.norm(), std::numeric_limits<T>::min());
      const T gradient_relative_linf =
        reconstruction_residual.template lpNorm<Eigen::Infinity>() /
        std::max<T>(
          full_gradient.template lpNorm<Eigen::Infinity>(),
          std::numeric_limits<T>::min());
      const T objective_relative =
        std::abs(full_objective - weighted_objective_sum) /
        std::max<T>(std::abs(full_objective), std::numeric_limits<T>::min());
      if (gradient_relative_l2 > T(1.0e-10) ||
          gradient_relative_linf > T(1.0e-10) ||
          objective_relative > T(1.0e-10))
        throw std::runtime_error(
          "Objective-gradient atomic reconstruction exceeds 1e-10");

      std::ofstream terms(term_path.c_str());
      if (!terms)
        throw std::runtime_error(
          std::string("Cannot write objective-gradient term audit: ") +
          term_path);
      terms
        << "term\toriginal_weight\tunit_objective\tweighted_objective"
        << "\tunit_gradient_l2\tunit_gradient_linf"
        << "\tweighted_gradient_l2\tweighted_gradient_linf\n"
        << std::setprecision(std::numeric_limits<T>::max_digits10);
      for (int term = 0; term < term_count; ++term)
      {
        const T unit_l2 = records[term].unit_gradient.norm();
        const T unit_linf =
          records[term].unit_gradient.template lpNorm<Eigen::Infinity>();
        terms
          << records[term].name << "\t"
          << records[term].original_weight << "\t"
          << records[term].unit_objective << "\t"
          << records[term].original_weight * records[term].unit_objective << "\t"
          << unit_l2 << "\t" << unit_linf << "\t"
          << std::abs(records[term].original_weight) * unit_l2 << "\t"
          << std::abs(records[term].original_weight) * unit_linf << "\n";
      }
      terms
        << "__direct_full_before__\t1\t" << full_objective_before << "\t"
        << full_objective_before << "\t" << full_gradient_before.norm() << "\t"
        << full_gradient_before.template lpNorm<Eigen::Infinity>() << "\t"
        << full_gradient_before.norm() << "\t"
        << full_gradient_before.template lpNorm<Eigen::Infinity>() << "\n"
        << "__direct_full__\t1\t" << full_objective << "\t"
        << full_objective << "\t" << full_gradient.norm() << "\t"
        << full_gradient.template lpNorm<Eigen::Infinity>() << "\t"
        << full_gradient.norm() << "\t"
        << full_gradient.template lpNorm<Eigen::Infinity>() << "\n"
        << "__weighted_sum__\t1\t" << weighted_objective_sum << "\t"
        << weighted_objective_sum << "\t" << weighted_sum.norm() << "\t"
        << weighted_sum.template lpNorm<Eigen::Infinity>() << "\t"
        << weighted_sum.norm() << "\t"
        << weighted_sum.template lpNorm<Eigen::Infinity>() << "\n";
      terms.close();
      if (!terms)
        throw std::runtime_error(
          std::string("Failed while writing objective-gradient term audit: ") +
          term_path);

      std::ofstream vectors(vector_path.c_str());
      if (!vectors)
        throw std::runtime_error(
          std::string("Cannot write objective-gradient vector audit: ") +
          vector_path);
      vectors << "coefficient_index\tfield\tfield_local_index";
      for (int term = 0; term < term_count; ++term)
        vectors << "\t" << records[term].name;
      vectors
        << "\tdirect_full\tweighted_sum\treconstruction_residual\n"
        << std::setprecision(std::numeric_limits<T>::max_digits10);
      const int field_size = P.size() / 3;
      for (int coefficient = 0; coefficient < P.size(); ++coefficient)
      {
        vectors << coefficient << "\t" << coefficient / field_size
                << "\t" << coefficient % field_size;
        for (int term = 0; term < term_count; ++term)
          vectors << "\t" << records[term].unit_gradient(coefficient);
        vectors << "\t" << full_gradient(coefficient)
                << "\t" << weighted_sum(coefficient)
                << "\t" << reconstruction_residual(coefficient) << "\n";
      }
      vectors.close();
      if (!vectors)
        throw std::runtime_error(
          std::string("Failed while writing objective-gradient vector audit: ") +
          vector_path);

      std::cout
        << "PIMFA_OBJECTIVE_GRADIENT_AUDIT"
        << " terms=" << term_count
        << " coefficients=" << P.size()
        << " full_objective=" << std::setprecision(17) << full_objective
        << " weighted_objective_sum=" << weighted_objective_sum
        << " objective_relative_mismatch=" << objective_relative
        << " restoration_objective_abs=" << restoration_objective_abs
        << " restoration_gradient_linf=" << restoration_gradient_linf
        << " gradient_relative_l2_mismatch=" << gradient_relative_l2
        << " gradient_relative_linf_mismatch=" << gradient_relative_linf
        << " vector_path=" << vector_path
        << " term_path=" << term_path
        << "\n";
    }

private:
    void initializePhysicsAndIntegralOperators()
    {
      bcx_physics_mask = nsPhysicsTimeMask<T>(
        input_BCx.domain.col(2),
        integral_config.physics_t_min,
        active_bcx_count);
      bcy_physics_mask = nsPhysicsTimeMask<T>(
        input_BCy.domain.col(2),
        integral_config.physics_t_min,
        active_bcy_count);
      physics_time_mask_active =
        active_bcx_count < static_cast<T>(input_BCx.npts) ||
        active_bcy_count < static_cast<T>(input_BCy.npts);

      if (integral_config.operatorsRequired())
      {
        if (N1.cols() != N2.cols() || N1.cols() != N3.cols())
          throw std::invalid_argument(
            "Integral objective requires identical u/v/p coefficient counts");
        if (N1.rows() != input.npts ||
            N1_x.rows() != input.npts ||
            N1_y.rows() != input.npts ||
            N1_t.rows() != input.npts)
          throw std::invalid_argument(
            "Integral objective requires full-grid MFA operators");

        MatrixType physical_coordinates(input.npts, 3);
        physical_coordinates = input.domain.leftCols(3);
        NSIntegralBalanceConfig<T> global_config = integral_config;
        global_config.parts_x = 1;
        global_config.parts_y = 1;
        if (integral_config.quadrature == "trapezoid")
        {
          integral_operators.reset(
            new NSIntegralBalanceOperators<T>(
              NSIntegralBalance<T>::build(
                physical_coordinates, integral_config)));
          global_integral_operators.reset(
            new NSIntegralBalanceOperators<T>(
              NSIntegralBalance<T>::build(
                physical_coordinates, global_config)));
        }
        else if (integral_config.quadrature == "gl4")
        {
          if (!input.is_structured() || input.ndom_pts().size() != 3 ||
              input.ndom_pts().prod() != input.npts)
            throw std::invalid_argument(
              "Knot-span GL4 balance requires a structured 3-D x-y-t grid");
          if (mfa_data.tmesh.tensor_prods.size() != 1 ||
              mfa_data2.tmesh.tensor_prods.size() != 1 ||
              mfa_data3.tmesh.tensor_prods.size() != 1)
            throw std::invalid_argument(
              "Knot-span GL4 balance requires one tensor product per field");

          const TensorProduct<T>& tensor_u =
            mfa_data.tmesh.tensor_prods[0];
          const TensorProduct<T>& tensor_v =
            mfa_data2.tmesh.tensor_prods[0];
          const TensorProduct<T>& tensor_p =
            mfa_data3.tmesh.tensor_prods[0];
          if ((tensor_u.nctrl_pts.array() !=
               tensor_v.nctrl_pts.array()).any() ||
              (tensor_u.nctrl_pts.array() !=
               tensor_p.nctrl_pts.array()).any())
            throw std::invalid_argument(
              "Knot-span GL4 balance requires identical u/v/p control lattices");
          const auto nonrational = [](const TensorProduct<T>& tensor)
          {
            return tensor.weights.size() == 0 ||
              (tensor.weights.size() == tensor.nctrl_pts.prod() &&
               tensor.weights.allFinite() &&
               (tensor.weights.array() == T(1)).all());
          };
          if (!nonrational(tensor_u) || !nonrational(tensor_v) ||
              !nonrational(tensor_p))
            throw std::invalid_argument(
              "Knot-span GL4 balance requires nonrational (unit-weight) fields");

          std::array<std::vector<T>, 3> knots;
          std::array<int, 3> degrees;
          std::array<T, 3> physical_lower;
          std::array<T, 3> physical_upper;
          for (int axis = 0; axis < 3; ++axis)
          {
            degrees[axis] = mfa_data.p(axis);
            knots[axis] = mfa_data.tmesh.all_knots[axis];
            physical_lower[axis] = input.domain.col(axis).minCoeff();
            physical_upper[axis] = input.domain.col(axis).maxCoeff();
            if (mfa_data2.p(axis) != degrees[axis] ||
                mfa_data3.p(axis) != degrees[axis] ||
                mfa_data2.tmesh.all_knots[axis] != knots[axis] ||
                mfa_data3.tmesh.all_knots[axis] != knots[axis])
              throw std::invalid_argument(
                "Knot-span GL4 balance requires identical u/v/p spline "
                "degrees and knot vectors");
          }

          // The direct coefficient evaluator maps parametric knots to each
          // physical axis affinely.  Fail closed if a future caller supplies
          // a nonlinear parameterization rather than silently integrating a
          // different physical field.
          for (int axis = 0; axis < 3; ++axis)
          {
            const std::vector<T>& parameter_axis =
              input.params->param_grid[axis];
            if (parameter_axis.size() !=
                static_cast<size_t>(input.ndom_pts(axis)))
              throw std::invalid_argument(
                "Knot-span GL4 balance parameter-axis size mismatch");
            const int stride = axis == 0 ? 1 :
              (axis == 1 ? input.ndom_pts(0) :
               input.ndom_pts(0) * input.ndom_pts(1));
            const T tolerance = T(256) *
              std::numeric_limits<T>::epsilon();
            for (int index = 0; index < input.ndom_pts(axis); ++index)
            {
              const T normalized =
                (input.domain(index * stride, axis) - physical_lower[axis]) /
                (physical_upper[axis] - physical_lower[axis]);
              if (!std::isfinite(normalized) ||
                  std::abs(parameter_axis[index] - normalized) > tolerance)
                throw std::invalid_argument(
                  "Knot-span GL4 balance requires affine physical "
                  "parameterization on every axis");
            }
          }

          const int spatial_stride =
            input.ndom_pts(0) * input.ndom_pts(1);
          std::vector<T> time_axis(input.ndom_pts(2));
          for (int time = 0; time < input.ndom_pts(2); ++time)
            time_axis[time] = input.domain(time * spatial_stride, 2);

          knot_span_integral_operators.reset(
            new NSKnotSpanGLBalanceOperators<T>(
              NSKnotSpanGLBalance<T>::build(
                knots, degrees, physical_lower, physical_upper,
                time_axis, integral_config, 4)));
          global_knot_span_integral_operators.reset(
            new NSKnotSpanGLBalanceOperators<T>(
              NSKnotSpanGLBalance<T>::build(
                knots, degrees, physical_lower, physical_upper,
                time_axis, global_config, 4)));
        }
        else
          throw std::logic_error(
            "Unsupported conservative quadrature escaped validation");
      }

      if (physics_time_mask_active || integral_config.operatorsRequired())
      {
        const bool strong_enabled =
          lambda_xmom > T(0) || lambda_ymom > T(0) || lambda_div > T(0);
        const char* method =
          strong_enabled && integral_config.enabled() ? "hybrid" :
          integral_config.enabled() ? "integral_only" : "strong_only";
        std::cout
          << std::setprecision(17)
          << "PIMFA_NS_INTEGRAL_CONFIG"
          << " method=" << method
          << " lambda_mass=" << integral_config.lambda_mass
          << " lambda_xmom=" << integral_config.lambda_xmom
          << " lambda_ymom=" << integral_config.lambda_ymom
          << " parts_x=" << integral_config.parts_x
          << " parts_y=" << integral_config.parts_y
          << " quadrature=" << integral_config.quadrature
          << " physics_t_min=" << integral_config.physics_t_min
          << " viscosity=" << integral_config.viscosity
          << " diagnostics=" << (integral_config.diagnostics ? 1 : 0)
          << " closure_check=" << (integral_config.check_closure ? 1 : 0)
          << " closure_tolerance=" << integral_config.closure_tolerance
          << " bc_x_active=" << active_bcx_count
          << " bc_y_active=" << active_bcy_count
          << " ic_active=" << input_IC.npts;
        if (integral_operators)
        {
          std::cout
            << " fit_rows=" << integral_operators->fit.rows()
            << " validation_rows=" << integral_operators->validation.rows()
            << " fit_nnz=" << integral_operators->fit.nonzeros()
            << " validation_nnz=" << integral_operators->validation.nonzeros()
            << " global_fit_rows=" << global_integral_operators->fit.rows()
            << " global_validation_rows="
            << global_integral_operators->validation.rows()
            << " global_fit_nnz=" << global_integral_operators->fit.nonzeros()
            << " global_validation_nnz="
            << global_integral_operators->validation.nonzeros()
            << " x_boundaries=";
          for (size_t i = 0;
               i < integral_operators->x_boundaries.size(); ++i)
            std::cout << (i ? "," : "")
                      << integral_operators->x_boundaries[i];
          std::cout << " y_boundaries=";
          for (size_t i = 0;
               i < integral_operators->y_boundaries.size(); ++i)
            std::cout << (i ? "," : "")
                      << integral_operators->y_boundaries[i];
        }
        else if (knot_span_integral_operators)
        {
          std::cout
            << " fit_rows=" << knot_span_integral_operators->fit.rows()
            << " validation_rows="
            << knot_span_integral_operators->validation.rows()
            << " fit_face_samples="
            << knot_span_integral_operators->fit.faceSampleCount()
            << " validation_face_samples="
            << knot_span_integral_operators->validation.faceSampleCount()
            << " global_fit_rows="
            << global_knot_span_integral_operators->fit.rows()
            << " global_validation_rows="
            << global_knot_span_integral_operators->validation.rows()
            << " global_fit_face_samples="
            << global_knot_span_integral_operators->fit.faceSampleCount()
            << " global_validation_face_samples="
            << global_knot_span_integral_operators->validation.faceSampleCount()
            << " internal_face_closure="
            << knot_span_integral_operators->fit.maximum_internal_face_closure
            << " x_boundaries=";
          for (size_t i = 0;
               i < knot_span_integral_operators->x_boundaries.size(); ++i)
            std::cout << (i ? "," : "")
                      << knot_span_integral_operators->x_boundaries[i];
          std::cout << " y_boundaries=";
          for (size_t i = 0;
               i < knot_span_integral_operators->y_boundaries.size(); ++i)
            std::cout << (i ? "," : "")
                      << knot_span_integral_operators->y_boundaries[i];
        }
        else
        {
          std::cout
            << " fit_rows=0 validation_rows=0 fit_nnz=0 validation_nnz=0"
            << " global_fit_rows=0 global_validation_rows=0"
            << " global_fit_nnz=0 global_validation_nnz=0";
        }
        std::cout << "\n";
      }
    }

    std::vector<int> loadRequiredBenchmarkRows()
    {
      const char* filename = std::getenv("MFA_PDE_ACTIVE_INDEX_FILE");
      if (!filename || !*filename)
        throw std::invalid_argument(
          "MFA_PREFILTERED_PDE_BENCHMARK requires MFA_PDE_ACTIVE_INDEX_FILE");
      std::ifstream stream(filename);
      if (!stream)
        throw std::runtime_error(
          std::string("Cannot open MFA_PDE_ACTIVE_INDEX_FILE: ") + filename);

      std::vector<int> rows;
      std::vector<unsigned char> seen(
        static_cast<size_t>(input.npts), static_cast<unsigned char>(0));
      long long index = -1;
      while (stream >> index)
      {
        if (index < 0 || index >= static_cast<long long>(input.npts))
          throw std::runtime_error(
            "Out-of-range row in MFA_PDE_ACTIVE_INDEX_FILE");
        const size_t position = static_cast<size_t>(index);
        if (!seen[position])
        {
          seen[position] = 1;
          rows.push_back(static_cast<int>(index));
        }
      }
      if (!stream.eof())
        throw std::runtime_error(
          "Malformed row in MFA_PDE_ACTIVE_INDEX_FILE");
      if (rows.empty())
        throw std::runtime_error(
          "No rows in MFA_PDE_ACTIVE_INDEX_FILE");
      if (!std::is_sorted(rows.begin(), rows.end()))
        throw std::runtime_error(
          "Prefiltered PDE benchmark requires sorted canonical row indices");
      return rows;
    }

    void compactCollocationMatrix(
        MFA_Data<T>& data,
        const std::vector<int>& rows,
        const VectorXi& derivative_orders,
        bool use_derivatives,
        SparseMatrixX<T>& matrix,
        SparseMatrixX<T>& transpose)
    {
      TensorProduct<T>& tensor = data.tmesh.tensor_prods[0];
      const int coefficient_count = tensor.nctrl_pts.prod();
      const int point_count = static_cast<int>(rows.size());
      matrix.resize(point_count, coefficient_count);
      transpose.resize(coefficient_count, point_count);

      const VectorXi q = data.p + VectorXi::Ones(dom_dim);
      const int basis_per_point = q.prod();
      transpose.reserve(VectorXi::Constant(point_count, basis_per_point));

      VectorXi control_strides = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; ++k)
        control_strides(k) =
          control_strides(k - 1) * tensor.nctrl_pts(k - 1);

      BasisFunInfo<T> basis_info(q);
      std::vector<std::vector<T> > basis(dom_dim);
      std::vector<std::vector<std::vector<T> > > derivative_basis(dom_dim);
      for (int k = 0; k < dom_dim; ++k)
      {
        basis[k].resize(q(k));
        derivative_basis[k].resize(3);
        for (int derivative = 0; derivative < 3; ++derivative)
          derivative_basis[k][derivative].resize(q(k));
      }

      VectorXi control_starts(dom_dim);
      VectorXi spans(dom_dim);
      const VectorXi control_patch = data.p + VectorXi::Ones(dom_dim);
      VectorX<T> parameter(input.dom_dim);

      for (int compact_row = 0; compact_row < point_count; ++compact_row)
      {
        input.pt_params(
          static_cast<size_t>(rows[compact_row]), parameter);
        for (int k = 0; k < dom_dim; ++k)
        {
          const int degree = data.p(k);
          const T value = parameter(k);
          spans(k) = data.tmesh.FindSpan(k, value);
          control_starts(k) =
            spans(k) - degree - tensor.knot_mins[k];
          if (use_derivatives)
            data.FastBasisFunsDers(
              k, value, spans(k), derivative_orders(k),
              derivative_basis[k], basis_info);
          else
            data.FastBasisFuns(
              k, value, spans(k), basis[k], basis_info);
        }

        VolIterator control_iterator(control_patch);
        while (!control_iterator.done())
        {
          int control_index = 0;
          T coefficient = T(1);
          for (int k = 0; k < dom_dim; ++k)
          {
            const int local_index = control_iterator.idx_dim(k);
            control_index +=
              (local_index + control_starts(k)) * control_strides(k);
            coefficient *= use_derivatives
              ? derivative_basis[k][derivative_orders(k)][local_index]
              : basis[k][local_index];
          }
          transpose.insertBackUncompressed(
            control_index, compact_row) = coefficient;
          control_iterator.incr_iter();
        }
      }
      transpose.makeCompressed();
      matrix = transpose.transpose();
      matrix.makeCompressed();
    }

    void initializePrefilteredPdeBenchmark()
    {
      const std::chrono::steady_clock::time_point started =
        std::chrono::steady_clock::now();
      prefiltered_pde_rows = loadRequiredBenchmarkRows();
      VectorXi derivative_orders = VectorXi::Zero(dom_dim);

      compactCollocationMatrix(
        mfa_data, prefiltered_pde_rows, derivative_orders, false, N1, N1t);
      derivative_orders << 1, 0, 0;
      compactCollocationMatrix(
        mfa_data, prefiltered_pde_rows, derivative_orders, true, N1_x, N1t_x);
      derivative_orders << 2, 0, 0;
      compactCollocationMatrix(
        mfa_data, prefiltered_pde_rows, derivative_orders, true, N1_xx, N1t_xx);
      derivative_orders << 0, 1, 0;
      compactCollocationMatrix(
        mfa_data, prefiltered_pde_rows, derivative_orders, true, N1_y, N1t_y);
      derivative_orders << 0, 2, 0;
      compactCollocationMatrix(
        mfa_data, prefiltered_pde_rows, derivative_orders, true, N1_yy, N1t_yy);
      derivative_orders << 0, 0, 1;
      compactCollocationMatrix(
        mfa_data, prefiltered_pde_rows, derivative_orders, true, N1_t, N1t_t);

      const T x_extent =
        input.domain.col(0).maxCoeff() - input.domain.col(0).minCoeff();
      const T y_extent =
        input.domain.col(1).maxCoeff() - input.domain.col(1).minCoeff();
      const T time_extent =
        input.domain.col(2).maxCoeff() - input.domain.col(2).minCoeff();
      if (x_extent <= T(0) || y_extent <= T(0) || time_extent <= T(0))
        throw std::runtime_error(
          "NS prefiltered benchmark requires positive x/y/t extents");
      const T scale_x = T(1) / x_extent;
      const T scale_y = T(1) / y_extent;
      const T scale_time = T(1) / time_extent;
      N1_x *= scale_x; N1t_x *= scale_x;
      N1_xx *= scale_x * scale_x; N1t_xx *= scale_x * scale_x;
      N1_y *= scale_y; N1t_y *= scale_y;
      N1_yy *= scale_y * scale_y; N1t_yy *= scale_y * scale_y;
      N1_t *= scale_time; N1t_t *= scale_time;

      const size_t nonzeros =
        static_cast<size_t>(N1.nonZeros() + N1_x.nonZeros() +
          N1_xx.nonZeros() + N1_y.nonZeros() + N1_yy.nonZeros() +
          N1_t.nonZeros());
      prefiltered_construction_seconds =
        std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      std::cout
        << "PIMFA_PREFILTERED_PDE case=ns active_rows="
        << prefiltered_pde_rows.size()
        << " input_rows=" << input.npts
        << " operator_matrices=6"
        << " operator_nnz=" << nonzeros
        << " construction_seconds="
        << std::setprecision(17) << prefiltered_construction_seconds
        << "\n";
    }

    void loadRowMask(
        const char* environment_name,
        VectorX<T>& mask,
        T& row_count,
        const char* label,
        bool default_all)
    {
      mask = default_all
        ? VectorX<T>::Ones(input.npts)
        : VectorX<T>::Zero(input.npts);
      row_count = default_all ? static_cast<T>(input.npts) : T(0);
      const char* filename = std::getenv(environment_name);
      if (!filename || !*filename)
      {
        if (!default_all)
          fmt::print("NS {} rows disabled ({} is unset)\n", label, environment_name);
        return;
      }

      std::ifstream stream(filename);
      if (!stream)
        throw std::runtime_error(
          std::string("Cannot open ") + environment_name + ": " + filename);

      mask.setZero();
      std::vector<unsigned char> seen(
        static_cast<size_t>(input.npts), static_cast<unsigned char>(0));
      long long index = -1;
      size_t count = 0;
      while (stream >> index)
      {
        if (index < 0 || index >= static_cast<long long>(input.npts))
          throw std::runtime_error(
            std::string("Out-of-range row in ") + environment_name);
        const size_t position = static_cast<size_t>(index);
        if (!seen[position])
        {
          seen[position] = 1;
          mask(static_cast<Eigen::Index>(position)) = T(1);
          ++count;
        }
      }
      if (!stream.eof())
        throw std::runtime_error(
          std::string("Malformed row index in ") + environment_name);
      if (count == 0)
        throw std::runtime_error(
          std::string("No rows in ") + environment_name);
      row_count = static_cast<T>(count);
      fmt::print(
        "NS {} rows: using {} of {} from {}\n",
        label, count, input.npts, filename);
    }

      void printSparseMatrixElements(const SparseMatrixX<T>& matrix, const std::string& matrixName) {
      int count = 0;
      for (int k = 0; k < matrix.outerSize(); ++k) {
        for (typename SparseMatrixX<T>::InnerIterator it(matrix, k); it; ++it) {
	  if (count >= 27) {
	    return; // Stop after printing 27 elements
	  }
	  int row = it.row();
	  int col = it.col();
	  T value = it.value();
	  std::cout << "(" << row << ", " << col << "): " << value << std::endl;
	  ++count;
        }
      }
    }

    void CollMatrixUnified( TensorIdx           t_idx,    // index of tensor product containing input points and control points
				    SparseMatrixX<T>&   N, // (output) collocation matrix
			    SparseMatrixX<T>&   Nt)  // (output) transpose of collocation matrix
    {
      TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
      assert(Nt.rows() == t.nctrl_pts.prod());
      assert(Nt.cols() == input.npts);

      clock_t fill_time = clock();

      // resize matrices in case number of control points changed
      const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
      t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
      t.weights.resize(t.ctrl_pts.rows());

      // control point strides
      VectorXi cps = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; k++)
	{
	  cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	}

      // Prep for evaluating basis functions
      VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
      BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
      vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
      for (int k = 0; k < dom_dim; k++)
	{
	  lB[k].resize(q[k]);
	}

      VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
      VectorXi spans(dom_dim);                                       // current knot span in each dimension
      VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

      // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
      int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
      Nt.reserve(VectorXi::Constant(Nt.cols(), bf_per_pt));

      vector<int> row_nnzs(Nt.rows(), 0);

      // Iterate thru every point in subvolume given by tensor
      VectorX<T> param(input.dom_dim);
      for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
	{
	  input_it.params(param);

	  // Compute basis functions at input point
	  for (auto k = 0; k < dom_dim; k++)
	    {
	      int p   = mfa_data.p(k);
	      T   u   = param(k);

	      spans[k] = mfa_data.tmesh.FindSpan(k, u);

	      ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

	      mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi);
	    }

	  // Compute matrix value and insert into Nt
	  VolIterator ctrl_vol_iter(nctrl_patch);
	  while (!ctrl_vol_iter.done())
	    {
	      int ctrl_idx_full = 0;
	      T coeff_prod = 1;
	      for (int k = 0; k < dom_dim; k++)
		{
		  int idx = ctrl_vol_iter.idx_dim(k);
		  ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
		  coeff_prod *= lB[k][idx]; //local basis function
		}

	      // increment number of entries in this row (for use in constructing N)
	      row_nnzs[ctrl_idx_full]++;

	      Nt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;
	      // FRAGILE command; reserve() must be called prior, **with vector signature**
	      // entries must be inserted only once, and entries within a column must be inserted
	      // in order of increasing row index.
	      // This is an internal SparseMatrix command and no consistency checks are done.
	      //
	      // Empirical testing indicates that insertBackUncompressed is faster than insert, which
	      // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
	      // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

	      ctrl_vol_iter.incr_iter();
	    }
	}

      Nt.makeCompressed();

      // Fill N directly as the transpose of Nt, avoiding a temporary sparse transpose.
      {
	// Must do this after Nt is in compressed mode!
	N.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	int* col_starts     = N.outerIndexPtr();
	int* row_ids        = N.innerIndexPtr();
	T*   tr_values      = N.valuePtr();
	int* col_inner_nnzs = N.innerNonZeroPtr();

	int* oip    = Nt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	int* iip    = Nt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	T*   values = Nt.valuePtr();

	for (int o = 0; o < Nt.cols(); o++) // loop through columns of Nt (rows of N)
	  {
	    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
	      {
		int row = o;                // row in N
		int col = iip[oip[o]+i];    // col in N
		row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		// col_nnzs tracks how many entries we have inserted into each col buffer
		// Since we loop through the rows of N in order, the row indices within
		//     each col buffer are strictly increasing
		col_inner_nnzs[col]++;
	      }
	  }

	N.makeCompressed();
      }

      fill_time = clock() - fill_time;
    }
    void CollMatrixUnified2( TensorIdx           t_idx,    // index of tensor product containing input points and control points
				    SparseMatrixX<T>&   N, // (output) collocation matrix
			    SparseMatrixX<T>&   Nt)  // (output) transpose of collocation matrix
    {
      TensorProduct<T>& t = mfa_data2.tmesh.tensor_prods[t_idx];
      assert(Nt.rows() == t.nctrl_pts.prod());
      assert(Nt.cols() == input.npts);

      clock_t fill_time = clock();

      // resize matrices in case number of control points changed
      const int           pt_dim  = mfa_data2.dim();        // dimensionality of current model
      t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
      t.weights.resize(t.ctrl_pts.rows());

      // control point strides
      VectorXi cps = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; k++)
	{
	  cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	}

      // Prep for evaluating basis functions
      VectorXi            q = mfa_data2.p + VectorXi::Ones(dom_dim);  // order of basis funs
      BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
      vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
      for (int k = 0; k < dom_dim; k++)
	{
	  lB[k].resize(q[k]);
	}

      VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
      VectorXi spans(dom_dim);                                       // current knot span in each dimension
      VectorXi nctrl_patch = mfa_data2.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

      // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
      int bf_per_pt = (mfa_data2.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
      Nt.reserve(VectorXi::Constant(Nt.cols(), bf_per_pt));

      vector<int> row_nnzs(Nt.rows(), 0);

      // Iterate thru every point in subvolume given by tensor
      VectorX<T> param(input.dom_dim);
      for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
	{
	  input_it.params(param);

	  // Compute basis functions at input point
	  for (auto k = 0; k < dom_dim; k++)
	    {
	      int p   = mfa_data2.p(k);
	      T   u   = param(k);

	      spans[k] = mfa_data2.tmesh.FindSpan(k, u);

	      ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

	      mfa_data2.FastBasisFuns(k, u, spans[k], lB[k], bfi);
	    }

	  // Compute matrix value and insert into Nt
	  VolIterator ctrl_vol_iter(nctrl_patch);
	  while (!ctrl_vol_iter.done())
	    {
	      int ctrl_idx_full = 0;
	      T coeff_prod = 1;
	      for (int k = 0; k < dom_dim; k++)
		{
		  int idx = ctrl_vol_iter.idx_dim(k);
		  ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
		  coeff_prod *= lB[k][idx]; //local basis function
		}

	      // increment number of entries in this row (for use in constructing N)
	      row_nnzs[ctrl_idx_full]++;

	      Nt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;
	      // FRAGILE command; reserve() must be called prior, **with vector signature**
	      // entries must be inserted only once, and entries within a column must be inserted
	      // in order of increasing row index.
	      // This is an internal SparseMatrix command and no consistency checks are done.
	      //
	      // Empirical testing indicates that insertBackUncompressed is faster than insert, which
	      // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
	      // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

	      ctrl_vol_iter.incr_iter();
	    }
	}

      Nt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


      // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
      {
	// Must do this after Nt is in compressed mode!
	N.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	int* col_starts     = N.outerIndexPtr();
	int* row_ids        = N.innerIndexPtr();
	T*   tr_values      = N.valuePtr();
	int* col_inner_nnzs = N.innerNonZeroPtr();

	int* oip    = Nt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	int* iip    = Nt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	T*   values = Nt.valuePtr();

	for (int o = 0; o < Nt.cols(); o++) // loop through columns of Nt (rows of N)
	  {
	    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
	      {
		int row = o;                // row in N
		int col = iip[oip[o]+i];    // col in N
		row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		// col_nnzs tracks how many entries we have inserted into each col buffer
		// Since we loop through the rows of N in order, the row indices within
		//     each col buffer are strictly increasing
		col_inner_nnzs[col]++;
	      }
	  }

	N.makeCompressed();
      }

      fill_time = clock() - fill_time;
    }


    void CollMatrixUnifiedDers( TensorIdx           t_idx,    // index of tensor product containing input points and control points
				VectorXi derivs, // derv with respect to which dom_min
                                SparseMatrixX<T>&   N, // (output) collocation matrix
                                SparseMatrixX<T>&   Nt)  // (output) transpose of collocation matrix
    {
      TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
      assert(Nt.rows() == t.nctrl_pts.prod());
      assert(Nt.cols() == input.npts);

      clock_t fill_time = clock();

      // resize matrices in case number of control points changed
      const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
      t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
      t.weights.resize(t.ctrl_pts.rows());

      // control point strides
      VectorXi cps = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; k++)
	{
	  cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	}

      // Prep for evaluating basis functions
      VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
      BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
      //For PDE-constrained, in case of dom_dim=2
      vector<vector<T>>  D_dim0(3);      //dom_dim=0
      vector<vector<T>>  D_dim1(3);      //dom_dim=1
      vector<vector<T>>  D_dim2(3);      //dom_dim=2
      for (int nd = 0; nd < 3; nd++)
	{
	  D_dim0[nd].resize(q[0]);
	  D_dim1[nd].resize(q[1]);
	  D_dim2[nd].resize(q[2]);
	  }

      VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
      VectorXi spans(dom_dim);                                       // current knot span in each dimension
      VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

      // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
      int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
      Nt.reserve(VectorXi::Constant(Nt.cols(), bf_per_pt));

      vector<int> row_nnzs(Nt.rows(), 0);

      // Iterate thru every point in subvolume given by tensor
      VectorX<T> param(input.dom_dim);
      for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
	{
	  input_it.params(param);

	  // Compute basis functions at input point
	  for (auto k = 0; k < dom_dim; k++)
	    {
	      int p   = mfa_data.p(k);
	      T   u   = param(k);

	      spans[k] = mfa_data.tmesh.FindSpan(k, u);

	      ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

	      //Nx=BxB ,Nt=BBt,Nxx=BxxB
	      int derivv=derivs(k);
	      if (k==0){
		mfa_data.FastBasisFunsDers(k, u, spans[k],derivv, D_dim0, bfi);
	      }
	      else if (k==1){
		mfa_data.FastBasisFunsDers(k, u, spans[k],derivv, D_dim1, bfi);
	      }
	      else if (k==2){
		mfa_data.FastBasisFunsDers(k, u, spans[k],derivv, D_dim2, bfi);
	      }

	    }
	  // Compute matrix value and insert into Nt
	  VolIterator ctrl_vol_iter(nctrl_patch);
	  while (!ctrl_vol_iter.done())
	    {
	      int ctrl_idx_full = 0;
	      T coeff_prod = 1;
	      for (int k = 0; k < dom_dim; k++)
		{
		  int idx = ctrl_vol_iter.idx_dim(k);
		  ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
		  int nderivs=derivs(k);
		  if (k==0){
		    coeff_prod *= D_dim0[nderivs][idx];
		  }
		  else if (k==1){
		    coeff_prod *= D_dim1[nderivs][idx];
		  }
		  else if (k==2){
		    coeff_prod *= D_dim2[nderivs][idx];
		  }
		}

	      // increment number of entries in this row (for use in constructing N)
	      row_nnzs[ctrl_idx_full]++;

	      Nt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;
	      // FRAGILE command; reserve() must be called prior, **with vector signature**
	      // entries must be inserted only once, and entries within a column must be inserted
	      // in order of increasing row index.
	      // This is an internal SparseMatrix command and no consistency checks are done.
	      //
	      // Empirical testing indicates that insertBackUncompressed is faster than insert, which
	      // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
	      // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

	      ctrl_vol_iter.incr_iter();
	    }
	}

      Nt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


      // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
      {
	// Must do this after Nt is in compressed mode!
	N.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	int* col_starts     = N.outerIndexPtr();
	int* row_ids        = N.innerIndexPtr();
	T*   tr_values      = N.valuePtr();
	int* col_inner_nnzs = N.innerNonZeroPtr();

	int* oip    = Nt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	int* iip    = Nt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	T*   values = Nt.valuePtr();

	for (int o = 0; o < Nt.cols(); o++) // loop through columns of Nt (rows of N)
	  {
	    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
	      {
		int row = o;                // row in N
		int col = iip[oip[o]+i];    // col in N
		row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		// col_nnzs tracks how many entries we have inserted into each col buffer
		// Since we loop through the rows of N in order, the row indices within
		//     each col buffer are strictly increasing
		col_inner_nnzs[col]++;
	      }
	  }

	N.makeCompressed();
      }

      fill_time = clock() - fill_time;
    }
    void CollMatrixUnifiedDers2( TensorIdx           t_idx,    // index of tensor product containing input points and control points
				VectorXi derivs, // derv with respect to which dom_min
                                SparseMatrixX<T>&   N, // (output) collocation matrix
                                SparseMatrixX<T>&   Nt)  // (output) transpose of collocation matrix
    {
      TensorProduct<T>& t = mfa_data2.tmesh.tensor_prods[t_idx];
      assert(Nt.rows() == t.nctrl_pts.prod());
      assert(Nt.cols() == input.npts);

      clock_t fill_time = clock();

      // resize matrices in case number of control points changed
      const int           pt_dim  = mfa_data2.dim();        // dimensionality of current model
      t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
      t.weights.resize(t.ctrl_pts.rows());

      // control point strides
      VectorXi cps = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; k++)
	{
	  cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	}

      // Prep for evaluating basis functions
      VectorXi            q = mfa_data2.p + VectorXi::Ones(dom_dim);  // order of basis funs
      BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
      //For PDE-constrained, in case of dom_dim=2
      vector<vector<T>>  D_dim0(3);      //dom_dim=0
      vector<vector<T>>  D_dim1(3);      //dom_dim=1
      vector<vector<T>>  D_dim2(3);      //dom_dim=2
      for (int nd = 0; nd < 3; nd++)
	{
	  D_dim0[nd].resize(q[0]);
	  D_dim1[nd].resize(q[1]);
	  D_dim2[nd].resize(q[2]);
	  }

      VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
      VectorXi spans(dom_dim);                                       // current knot span in each dimension
      VectorXi nctrl_patch = mfa_data2.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

      // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
      int bf_per_pt = (mfa_data2.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
      Nt.reserve(VectorXi::Constant(Nt.cols(), bf_per_pt));

      vector<int> row_nnzs(Nt.rows(), 0);

      // Iterate thru every point in subvolume given by tensor
      VectorX<T> param(input.dom_dim);
      for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
	{
	  input_it.params(param);

	  // Compute basis functions at input point
	  for (auto k = 0; k < dom_dim; k++)
	    {
	      int p   = mfa_data2.p(k);
	      T   u   = param(k);

	      spans[k] = mfa_data2.tmesh.FindSpan(k, u);

	      ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

	      //Nx=BxB ,Nt=BBt,Nxx=BxxB
	      int derivv=derivs(k);
	      if (k==0){
		mfa_data2.FastBasisFunsDers(k, u, spans[k],derivv, D_dim0, bfi);
	      }
	      else if (k==1){
		mfa_data2.FastBasisFunsDers(k, u, spans[k],derivv, D_dim1, bfi);
	      }
	      else if (k==2){
		mfa_data2.FastBasisFunsDers(k, u, spans[k],derivv, D_dim2, bfi);
	      }

	    }
	  // Compute matrix value and insert into Nt
	  VolIterator ctrl_vol_iter(nctrl_patch);
	  while (!ctrl_vol_iter.done())
	    {
	      int ctrl_idx_full = 0;
	      T coeff_prod = 1;
	      for (int k = 0; k < dom_dim; k++)
		{
		  int idx = ctrl_vol_iter.idx_dim(k);
		  ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
		  int nderivs=derivs(k);
		  if (k==0){
		    coeff_prod *= D_dim0[nderivs][idx];
		  }
		  else if (k==1){
		    coeff_prod *= D_dim1[nderivs][idx];
		  }
		  else if (k==2){
		    coeff_prod *= D_dim2[nderivs][idx];
		  }
		}

	      // increment number of entries in this row (for use in constructing N)
	      row_nnzs[ctrl_idx_full]++;

	      Nt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;
	      // FRAGILE command; reserve() must be called prior, **with vector signature**
	      // entries must be inserted only once, and entries within a column must be inserted
	      // in order of increasing row index.
	      // This is an internal SparseMatrix command and no consistency checks are done.
	      //
	      // Empirical testing indicates that insertBackUncompressed is faster than insert, which
	      // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
	      // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

	      ctrl_vol_iter.incr_iter();
	    }
	}

      Nt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


      // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
      {
	// Must do this after Nt is in compressed mode!
	N.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	int* col_starts     = N.outerIndexPtr();
	int* row_ids        = N.innerIndexPtr();
	T*   tr_values      = N.valuePtr();
	int* col_inner_nnzs = N.innerNonZeroPtr();

	int* oip    = Nt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	int* iip    = Nt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	T*   values = Nt.valuePtr();

	for (int o = 0; o < Nt.cols(); o++) // loop through columns of Nt (rows of N)
	  {
	    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
	      {
		int row = o;                // row in N
		int col = iip[oip[o]+i];    // col in N
		row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		// col_nnzs tracks how many entries we have inserted into each col buffer
		// Since we loop through the rows of N in order, the row indices within
		//     each col buffer are strictly increasing
		col_inner_nnzs[col]++;
	      }
	  }

	N.makeCompressed();
      }

      fill_time = clock() - fill_time;
    }
    void ICMatrix(
		    TensorIdx               t_idx,
		    SparseMatrixX<T>&       IC, // (output) BC-constrained matrix
		    SparseMatrixX<T>&       ICt) //(output) BC-constrained matrix of transpose
    {
      clock_t fill_time = clock();

      TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];


      // resize matrices in case number of control points changed
      const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
      t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
      t.weights.resize(t.ctrl_pts.rows());

      // control point strides
      VectorXi cps = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; k++)
	{
	  cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	}

      // Prep for evaluating basis functions
      VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
      BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
      vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
      for (int k = 0; k < dom_dim; k++)
	{
	  lB[k].resize(q[k]);
	}

      VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
      VectorXi spans(dom_dim);                                       // current knot span in each dimension
      VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension  //p+1

      // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
      int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
      ICt.reserve(VectorXi::Constant(ICt.cols(), bf_per_pt)); //Bt.cols=2001, bf_per_pt=16 (nonzero values)

      vector<int> row_nnzs(ICt.rows(), 0); //Bt.rows(): # ofcontrol points

      // Iterate thru every point in subvolume given by tensor
      VectorX<T> param(input_IC.dom_dim);
      for (auto input_it = input_IC.begin(), input_end = input_IC.end(); input_it != input_end; ++input_it) //main loop, every row
	{

	  input_it.params(param); //get parametric representation at the point,and  param is vector, iteration from the first point : input_it

	  // Compute basis functions at input point
	  for (auto k = 0; k < dom_dim; k++)
	    {
	      int p   = mfa_data.p(k); //polynomial degree =p
	      T   u   = param(k);   //compute u and v
	      spans[k] = mfa_data.tmesh.FindSpan(k, u); // compute not whole, but we find correspoinding(relavant) span to the point, because most of the basis functions for a specific point  are zero./ which basis functions are computed

	      ctrl_starts(k) = spans[k] - p - t.knot_mins[k]; //indexing

	      mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi); // compute nonzero basis function in 1D [k], => store results in lb[k]
	    }

	  // Compute matrix value and insert into Nt
	  VolIterator ctrl_vol_iter(nctrl_patch); //get index for the patch that are relavant to the basis functions --> That is, the points in a small squre are indexed. nctrl_patch=p+1=4
	  while (!ctrl_vol_iter.done())
	    {
	      int ctrl_idx_full = 0;
	      T coeff_prod = 1;
	      for (int k = 0; k < dom_dim; k++)
		{
		  int idx = ctrl_vol_iter.idx_dim(k);
		  ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k); //indexing
		  coeff_prod *= lB[k][idx];

		}

	      // increment number of entries in this row (for use in constructing N)
	      row_nnzs[ctrl_idx_full]++;

	      ICt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;

	      // FRAGILE command; reserve() must be called prior, **with vector signature**
	      // entries must be inserted only once, and entries within a column must be inserted
	      // in order of increasing row index.
	      // This is an internal SparseMatrix command and no consistency checks are done.
	      //
	      // Empirical testing indicates that insertBackUncompressed is faster than insert, which
	      // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
	      // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

	      ctrl_vol_iter.incr_iter();
	    }
	}

      ICt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


      // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
      {
	// Must do this after Nt is in compressed mode!
	IC.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	int* col_starts     = IC.outerIndexPtr();   //SparsMatrix outerIndexPtr, innerIndexPtr, valuePtr are inline functions, so the functions are used here, not re-call process (see inline functions in c++ tutorial)
	int* row_ids        = IC.innerIndexPtr();
	T*   tr_values      = IC.valuePtr();
	int* col_inner_nnzs = IC.innerNonZeroPtr();

	int* oip    = ICt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	int* iip    = ICt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	T*   values = ICt.valuePtr();

	for (int o = 0; o < ICt.cols(); o++) // loop through columns of Nt (rows of N)
	  {
	    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
	      {
		int row = o;                // row in N
		int col = iip[oip[o]+i];    // col in N
		row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		// col_nnzs tracks how many entries we have inserted into each col buffer
		// Since we loop through the rows of N in order, the row indices within
		//     each col buffer are strictly increasing
		col_inner_nnzs[col]++;
	      }
	  }

	IC.makeCompressed();
      }




      fill_time = clock() - fill_time;
    }
    void BCxMatrix(
		    TensorIdx               t_idx,
		    SparseMatrixX<T>&       B, // (output) BC-constrained matrix
		    SparseMatrixX<T>&       Bt) //(output) BC-constrained matrix of transpose
    {
      clock_t fill_time = clock();

      TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];


      // resize matrices in case number of control points changed
      const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
      t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
      t.weights.resize(t.ctrl_pts.rows());

      // control point strides
      VectorXi cps = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; k++)
	{
	  cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	}

      // Prep for evaluating basis functions
      VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
      BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
      vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
      for (int k = 0; k < dom_dim; k++)
	{
	  lB[k].resize(q[k]);
	}

      VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
      VectorXi spans(dom_dim);                                       // current knot span in each dimension
      VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension  //p+1

      // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
      int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
      Bt.reserve(VectorXi::Constant(Bt.cols(), bf_per_pt)); //Bt.cols=2001, bf_per_pt=16 (nonzero values)

      vector<int> row_nnzs(Bt.rows(), 0); //Bt.rows(): # ofcontrol points

      // Iterate thru every point in subvolume given by tensor
      VectorX<T> param(input_BCx.dom_dim);
      for (auto input_it = input_BCx.begin(), input_end = input_BCx.end(); input_it != input_end; ++input_it) //main loop, every row
	{

	  input_it.params(param); //get parametric representation at the point,and  param is vector, iteration from the first point : input_it

	  // Compute basis functions at input point
	  for (auto k = 0; k < dom_dim; k++)
	    {
	      int p   = mfa_data.p(k); //polynomial degree =p
	      T   u   = param(k);   //compute u and v
	      spans[k] = mfa_data.tmesh.FindSpan(k, u); // compute not whole, but we find correspoinding(relavant) span to the point, because most of the basis functions for a specific point  are zero./ which basis functions are computed

	      ctrl_starts(k) = spans[k] - p - t.knot_mins[k]; //indexing

	      mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi); // compute nonzero basis function in 1D [k], => store results in lb[k]
	    }

	  // Compute matrix value and insert into Nt
	  VolIterator ctrl_vol_iter(nctrl_patch); //get index for the patch that are relavant to the basis functions --> That is, the points in a small squre are indexed. nctrl_patch=p+1=4
	  while (!ctrl_vol_iter.done())
	    {
	      int ctrl_idx_full = 0;
	      T coeff_prod = 1;
	      for (int k = 0; k < dom_dim; k++)
		{
		  int idx = ctrl_vol_iter.idx_dim(k);
		  ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k); //indexing
		  coeff_prod *= lB[k][idx];

		}

	      // increment number of entries in this row (for use in constructing N)
	      row_nnzs[ctrl_idx_full]++;

	      Bt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;

	      // FRAGILE command; reserve() must be called prior, **with vector signature**
	      // entries must be inserted only once, and entries within a column must be inserted
	      // in order of increasing row index.
	      // This is an internal SparseMatrix command and no consistency checks are done.
	      //
	      // Empirical testing indicates that insertBackUncompressed is faster than insert, which
	      // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
	      // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

	      ctrl_vol_iter.incr_iter();
	    }
	}


      Bt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


      // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
      {
	// Must do this after Nt is in compressed mode!
	B.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	int* col_starts     = B.outerIndexPtr();   //SparsMatrix outerIndexPtr, innerIndexPtr, valuePtr are inline functions, so the functions are used here, not re-call process (see inline functions in c++ tutorial)
	int* row_ids        = B.innerIndexPtr();
	T*   tr_values      = B.valuePtr();
	int* col_inner_nnzs = B.innerNonZeroPtr();

	int* oip    = Bt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	int* iip    = Bt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	T*   values = Bt.valuePtr();

	for (int o = 0; o < Bt.cols(); o++) // loop through columns of Nt (rows of N)
	  {
	    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
	      {
		int row = o;                // row in N
		int col = iip[oip[o]+i];    // col in N
		row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		// col_nnzs tracks how many entries we have inserted into each col buffer
		// Since we loop through the rows of N in order, the row indices within
		//     each col buffer are strictly increasing
		col_inner_nnzs[col]++;
	      }
	  }

	B.makeCompressed();
      }




      fill_time = clock() - fill_time;
    }

    void BCyMatrix(
		    TensorIdx               t_idx,
		    SparseMatrixX<T>&       B, // (output) BC-constrained matrix
		    SparseMatrixX<T>&       Bt) //(output) BC-constrained matrix of transpose
    {
      clock_t fill_time = clock();

      TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];


      // resize matrices in case number of control points changed
      const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
      t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
      t.weights.resize(t.ctrl_pts.rows());

      // control point strides
      VectorXi cps = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; k++)
	{
	  cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	}

      // Prep for evaluating basis functions
      VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
      BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
      vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
      for (int k = 0; k < dom_dim; k++)
	{
	  lB[k].resize(q[k]);
	}

      VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
      VectorXi spans(dom_dim);                                       // current knot span in each dimension
      VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension  //p+1

      // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
      int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
      Bt.reserve(VectorXi::Constant(Bt.cols(), bf_per_pt)); //Bt.cols=2001, bf_per_pt=16 (nonzero values)

      vector<int> row_nnzs(Bt.rows(), 0); //Bt.rows(): # ofcontrol points

      // Iterate thru every point in subvolume given by tensor
      VectorX<T> param(input_BCy.dom_dim);
      for (auto input_it = input_BCy.begin(), input_end = input_BCy.end(); input_it != input_end; ++input_it) //main loop, every row
	{

	  input_it.params(param); //get parametric representation at the point,and  param is vector, iteration from the first point : input_it

	  // Compute basis functions at input point
	  for (auto k = 0; k < dom_dim; k++)
	    {
	      int p   = mfa_data.p(k); //polynomial degree =p
	      T   u   = param(k);   //compute u and v
	      spans[k] = mfa_data.tmesh.FindSpan(k, u); // compute not whole, but we find correspoinding(relavant) span to the point, because most of the basis functions for a specific point  are zero./ which basis functions are computed

	      ctrl_starts(k) = spans[k] - p - t.knot_mins[k]; //indexing

	      mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi); // compute nonzero basis function in 1D [k], => store results in lb[k]
	    }

	  // Compute matrix value and insert into Nt
	  VolIterator ctrl_vol_iter(nctrl_patch); //get index for the patch that are relavant to the basis functions --> That is, the points in a small squre are indexed. nctrl_patch=p+1=4
	  while (!ctrl_vol_iter.done())
	    {
	      int ctrl_idx_full = 0;
	      T coeff_prod = 1;
	      for (int k = 0; k < dom_dim; k++)
		{
		  int idx = ctrl_vol_iter.idx_dim(k);
		  ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k); //indexing
		  coeff_prod *= lB[k][idx];

		}

	      // increment number of entries in this row (for use in constructing N)
	      row_nnzs[ctrl_idx_full]++;

	      Bt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;

	      // FRAGILE command; reserve() must be called prior, **with vector signature**
	      // entries must be inserted only once, and entries within a column must be inserted
	      // in order of increasing row index.
	      // This is an internal SparseMatrix command and no consistency checks are done.
	      //
	      // Empirical testing indicates that insertBackUncompressed is faster than insert, which
	      // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
	      // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

	      ctrl_vol_iter.incr_iter();
	    }
	}


      Bt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


      // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
      {
	// Must do this after Nt is in compressed mode!
	B.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	int* col_starts     = B.outerIndexPtr();   //SparsMatrix outerIndexPtr, innerIndexPtr, valuePtr are inline functions, so the functions are used here, not re-call process (see inline functions in c++ tutorial)
	int* row_ids        = B.innerIndexPtr();
	T*   tr_values      = B.valuePtr();
	int* col_inner_nnzs = B.innerNonZeroPtr();

	int* oip    = Bt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	int* iip    = Bt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	T*   values = Bt.valuePtr();

	for (int o = 0; o < Bt.cols(); o++) // loop through columns of Nt (rows of N)
	  {
	    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
	      {
		int row = o;                // row in N
		int col = iip[oip[o]+i];    // col in N
		row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		// col_nnzs tracks how many entries we have inserted into each col buffer
		// Since we loop through the rows of N in order, the row indices within
		//     each col buffer are strictly increasing
		col_inner_nnzs[col]++;
	      }
	  }

	B.makeCompressed();
      }




      fill_time = clock() - fill_time;
    }

};

}

template <typename T>                                       // float or double
class NewKnots;

template <typename T>                                       // float or double
struct MFA;

namespace mfa
{
    template <typename T>                                   // float or double
    class Encoder
    {
        enum class ConsType
        {
            MFA_NO_CONSTRAINT,
            MFA_LEFT_ONLY_CONSTRAINT,
            MFA_RIGHT_ONLY_CONSTRAINT,
            MFA_BOTH_CONSTRAINT
        };

    private:

        template <typename>
        friend class NewKnots;

        MFA_Data<T>&        mfa_data;                       // the mfa data model
        MFA_Data<T>&        mfa_data2;                       // the mfa data model
        MFA_Data<T>&        mfa_data3;                       // the mfa data model
        int                 dom_dim;                        // domain dimension of mfa_data
        int                 verbose;                        // output level
        const PointSet<T>&  input;                          // input points
        const PointSet<T>&  input_IC;                       // input points for IC
        const PointSet<T>&  input_BC;                       // input points for BC
        const PointSet<T>&  input_BCx;                       // input points for BC
        const PointSet<T>&  input_BCy;                       // input points for BC
        size_t              max_num_curves;                 // max num. curves per dimension to check in curve version
    public:
        Encoder(MFA_Data<T>&        mfa_data_,              // MFA data model
		MFA_Data<T>&        mfa_data_2,
		MFA_Data<T>&        mfa_data_3,
		const PointSet<T>&  input_,                 // input points
                const PointSet<T>&  input_IC_,              // input_IC points
		const PointSet<T>&  input_BC_,                       // input points for BC
                const PointSet<T>&  input_BCx_,              // input_BC points
                const PointSet<T>&  input_BCy_,              // input_BC points
                int                 verbose_) :             // debug level
            mfa_data(mfa_data_),
            mfa_data2(mfa_data_2),
            mfa_data3(mfa_data_3),
            dom_dim(mfa_data.dom_dim),
            verbose(verbose_),
            input(input_),
            input_IC(input_IC_),
            input_BC(input_BC_),
            input_BCx(input_BCx_),
            input_BCy(input_BCy_),
            max_num_curves(1.0e5)                           // max num. curves to check in one dimension of curve version
        { }

        ~Encoder() {}

        // approximate a NURBS hypervolume of arbitrary dimension for a given input data set
        // n-d version of algorithm 9.7, Piegl & Tiller (P&T) p. 422
        // output control points can be specified by caller, does not have to be those in the tmesh
        // the output ctrl_pts are resized by this function;  caller need not resize them
        void Encode(const VectorXi& nctrl_pts,              // number of control points in each dim.
                    MatrixX<T>&     ctrl_pts,               // (output) control points
                    VectorX<T>&     weights,                // (output) weights
                    bool            weighted = true)        // solve for and use weights
        {
            // check quantities
            if (mfa_data.p.size() != input.ndom_pts().size())
            {
                fprintf(stderr, "Error: Encode() size of p must equal size of ndom_pts\n");
                exit(1);
            }
            for (size_t i = 0; i < mfa_data.p.size(); i++)
            {
                if (nctrl_pts(i) <= mfa_data.p(i))
                {
                    fprintf(stderr, "Error: Encode() number of control points in dimension %ld "
                            "must be at least p + 1 for dimension %ld\n", i, i);
                    exit(1);
                }
                if (nctrl_pts(i) > input.ndom_pts(i))
                {
                    fprintf(stderr, "Warning: Encode() number of control points (%d) in dimension %ld "
                            "exceeds number of input data points (%d) in dimension %ld.\n", nctrl_pts(i), i, input.ndom_pts(i), i);
                }
            }

            int      ndims  = input.ndom_pts().size();          // number of domain dimensions
            size_t   cs     = 1;                            // stride for input points in curve in cur. dim
            int      pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;// control point dimensonality
            // resize matrices in case number of control points changed
            ctrl_pts.resize(nctrl_pts.prod(), pt_dim);
            weights.resize(ctrl_pts.rows());

            // resize basis function matrices and initialize to 0; will only fill nonzeros later
            mfa_data.N.resize(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
                mfa_data.N[k] = MatrixX<T>::Zero(input.ndom_pts(k), nctrl_pts(k));

            // 2 buffers of temporary control points
            // double buffer needed to write output curves of current dim without changing its input pts
            // temporary control points need to begin with size as many as the input domain points
            // except for the first dimension, which can be the correct number of control points
            // because the input domain points are converted to control points one dimension at a time
            // TODO: need to find a more space-efficient way
            size_t tot_ntemp_ctrl = 1;
            for (size_t k = 0; k < ndims; k++)
                tot_ntemp_ctrl *= (k == 0 ? nctrl_pts(k) : input.ndom_pts(k));
            MatrixX<T> temp_ctrl0 = MatrixX<T>::Zero(tot_ntemp_ctrl, pt_dim);
            MatrixX<T> temp_ctrl1 = MatrixX<T>::Zero(tot_ntemp_ctrl, pt_dim);

            VectorXi ntemp_ctrl = input.ndom_pts();     // current num of temp control pts in each dim

            for (size_t k = 0; k < ndims; k++)      // for all domain dimensions
            {
                // number of curves in this dimension
                size_t ncurves;
                ncurves = 1;
                for (int i = 0; i < ndims; i++)
                {
                    if (i < k)
                        ncurves *= nctrl_pts(i);
                    else if (i > k)
                        ncurves *= input.ndom_pts(i);
                    // NB: current dimension contributes no curves, hence no i == k case
                }

                // compute local version of co
                vector<size_t> co(ncurves);                     // starting curve points in current dim.
                vector<size_t> to(ncurves);                     // starting control points in current dim.
                co[0]      = 0;
                to[0]      = 0;
                size_t coo = 0;                                 // co at start of contiguous sequence
                size_t too = 0;                                 // to at start of contiguous sequence

                // TODO: allocate P and R once for all curves in an EncodeInfo struct similar to DecodeInfo
                for (auto j = 1; j < ncurves; j++)
                {
                    if (j % cs)
                    {
                        co[j] = co[j - 1] + 1;
                        to[j] = to[j - 1] + 1;
                    }
                    else
                    {
                        co[j] = coo + cs * ntemp_ctrl(k);
                        coo   = co[j];
                        to[j] = too + cs * nctrl_pts(k);
                        too   = to[j];
                    }
                }

                // N is a matrix of m (# input points) x n (# control points) basis function coefficients
                //  _                              _
                // |  N_0(u[0])   ... N_n-1(u[0])   |
                // |     ...      ...      ...      |
                // |  N_0(u[m-1]) ... N_n-1(u[m-1]) |
                //  -                              -
                // TODO: N is going to be very sparse when it is large: switch to sparse representation
                // N has semibandwidth < p  nonzero entries across diagonal

                for (int i = 0; i < mfa_data.N[k].rows(); i++)
                {
                    int span = mfa_data.tmesh.FindSpan(k, input.params->param_grid[k][i], nctrl_pts(k));

#ifndef MFA_TMESH   // original version for one tensor product

                    mfa_data.OrigBasisFuns(k, input.params->param_grid[k][i], span, mfa_data.N[k], i);

#else               // tmesh version

                    mfa_data.BasisFuns(k, input.params->param_grid[k][i], span, mfa_data.N[k], i);

#endif
                }


                // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
                // NtN has semibandwidth < p + 1 nonzero entries across diagonal
                MatrixX<T> NtN  = mfa_data.N[k].transpose() * mfa_data.N[k];


#ifdef MFA_TBB  // TBB version
                static affinity_partitioner ap;
                parallel_for (blocked_range<size_t>(0, ncurves), [&] (blocked_range<size_t>& r)
                {
                    for (auto j = r.begin(); j < r.end(); j++)        // for all the curves in this dimension
                    {

                        // R is the right hand side needed for solving NtN * P = R
                        MatrixX<T> R(mfa_data.N[k].cols(), pt_dim);
                        // P are the unknown control points and the solution to NtN * P = R
                        // NtN is positive definite -> do not need pivoting
                        // TODO: use a common representation for P and ctrl_pts to avoid copying
                        MatrixX<T> P(mfa_data.N[k].cols(), pt_dim);
                        // compute the one curve of control points
                        CtrlCurve(mfa_data.N[k], NtN, R, P, k, co[j], cs, to[j], temp_ctrl0, temp_ctrl1, -1, ctrl_pts, weights, weighted);
                    }   // curves in this dimension
                }, ap);

#endif              // end TBB version

#if defined(MFA_SERIAL) || defined(MFA_KOKKOS)
                // R is the right hand side needed for solving NtN * P = R
                MatrixX<T> R(mfa_data.N[k].cols(), pt_dim);

                // P are the unknown control points and the solution to NtN * P = R
                // NtN is positive definite -> do not need pivoting
                // TODO: use a common representation for P and ctrl_pts to avoid copying
                MatrixX<T> P(mfa_data.N[k].cols(), pt_dim);

                // encode curves in this dimension
                for (size_t j = 0; j < ncurves; j++)
                {
                    // print progress
                    {
                        if (j > 0 && j > 100 && j % (ncurves / 100) == 0)
                            fprintf(stderr, "\r dimension %ld: %.0f %% encoded (%ld out of %ld curves)",
                                    k, (T)j / (T)ncurves * 100, j, ncurves);
                    }

                    // compute the one curve of control points
                    CtrlCurve(mfa_data.N[k], NtN, R, P, k, co[j], cs, to[j], temp_ctrl0, temp_ctrl1, j, ctrl_pts, weights, weighted);
                }

#endif          // end serial version

                // adjust offsets and strides for next dimension
                ntemp_ctrl(k) = nctrl_pts(k);
                cs *= ntemp_ctrl(k);

                // print progress
                    fprintf(stderr, "\r dimension %ld: 100%% encoded                                       \n", k);
            }  // domain dimensions

        }

      void ICBCMatrix(
		    TensorIdx               t_idx,
		    SparseMatrixX<T>&       B, // (output) BC-constrained matrix
		    SparseMatrixX<T>&       Bt) //(output) BC-constrained matrix of transpose
      {
	clock_t fill_time = clock();

	TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];


	// resize matrices in case number of control points changed
	const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
	t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
	t.weights.resize(t.ctrl_pts.rows());

	// control point strides
	VectorXi cps = VectorXi::Ones(dom_dim);
	for (int k = 1; k < dom_dim; k++)
	  {
	    cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	  }

	// Prep for evaluating basis functions
	VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
	BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
	vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
	for (int k = 0; k < dom_dim; k++)
	  {
	    lB[k].resize(q[k]);
	  }

	VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
	VectorXi spans(dom_dim);                                       // current knot span in each dimension
	VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension  //p+1

	// Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
	int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
	Bt.reserve(VectorXi::Constant(Bt.cols(), bf_per_pt)); //Bt.cols=2001, bf_per_pt=16 (nonzero values)

	vector<int> row_nnzs(Bt.rows(), 0); //Bt.rows(): # ofcontrol points

	// Iterate thru every point in subvolume given by tensor
	VectorX<T> param(input_IC.dom_dim);
	for (auto input_it = input_IC.begin(), input_end = input_IC.end(); input_it != input_end; ++input_it) //main loop, every row
	  {

	    input_it.params(param); //get parametric representation at the point,and  param is vector, iteration from the first point : input_it

	    // Compute basis functions at input point
	    for (auto k = 0; k < dom_dim; k++)
	      {
                int p   = mfa_data.p(k); //polynomial degree =p
                T   u   = param(k);   //compute u and v
                spans[k] = mfa_data.tmesh.FindSpan(k, u); // compute not whole, but we find correspoinding(relavant) span to the point, because most of the basis functions for a specific point  are zero./ which basis functions are computed

                ctrl_starts(k) = spans[k] - p - t.knot_mins[k]; //indexing

                mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi); // compute nonzero basis function in 1D [k], => store results in lb[k]
	      }

	    // Compute matrix value and insert into Nt
	    VolIterator ctrl_vol_iter(nctrl_patch); //get index for the patch that are relavant to the basis functions --> That is, the points in a small squre are indexed. nctrl_patch=p+1=4

	    while (!ctrl_vol_iter.done())
	      {
                int ctrl_idx_full = 0;
                T coeff_prod = 1;
                for (int k = 0; k < dom_dim; k++)
		  {
                    int idx = ctrl_vol_iter.idx_dim(k);
                    ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k); //indexing
                    coeff_prod *= lB[k][idx];

		  }

                // increment number of entries in this row (for use in constructing N)
                row_nnzs[ctrl_idx_full]++;

                Bt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;

                // FRAGILE command; reserve() must be called prior, **with vector signature**
                // entries must be inserted only once, and entries within a column must be inserted
                // in order of increasing row index.
                // This is an internal SparseMatrix command and no consistency checks are done.
                //
                // Empirical testing indicates that insertBackUncompressed is faster than insert, which
                // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
                // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

                ctrl_vol_iter.incr_iter();
	      }
	  }

	Bt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


	// Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
	{
	  // Must do this after Nt is in compressed mode!
	  B.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	  int* col_starts     = B.outerIndexPtr();   //SparsMatrix outerIndexPtr, innerIndexPtr, valuePtr are inline functions, so the functions are used here, not re-call process (see inline functions in c++ tutorial)
	  int* row_ids        = B.innerIndexPtr();
	  T*   tr_values      = B.valuePtr();
	  int* col_inner_nnzs = B.innerNonZeroPtr();

	  int* oip    = Bt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	  int* iip    = Bt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	  T*   values = Bt.valuePtr();

	  for (int o = 0; o < Bt.cols(); o++) // loop through columns of Nt (rows of N)
	    {
	      for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
                {
		  int row = o;                // row in N
		  int col = iip[oip[o]+i];    // col in N
		  row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		  tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		  // col_nnzs tracks how many entries we have inserted into each col buffer
		  // Since we loop through the rows of N in order, the row indices within
		  //     each col buffer are strictly increasing
		  col_inner_nnzs[col]++;
                }
	    }

	  B.makeCompressed();
	}




        fill_time = clock() - fill_time;
      }

        void BCMatrix(
            TensorIdx               t_idx,
            SparseMatrixX<T>&       B, // (output) BC-constrained matrix
            SparseMatrixX<T>&       Bt) //(output) BC-constrained matrix of transpose
        {
            clock_t fill_time = clock();
            if (verbose)
            cerr << "constraining BC on the matrix N..." << endl;

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];


            // resize matrices in case number of control points changed
            const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());

            // control point strides
            VectorXi cps = VectorXi::Ones(dom_dim);
            for (int k = 1; k < dom_dim; k++)
            {
                cps(k) = cps(k-1) * t.nctrl_pts(k-1);
            }

            // Prep for evaluating basis functions
            VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
            BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
            vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
            for (int k = 0; k < dom_dim; k++)
            {
                lB[k].resize(q[k]);
            }

            VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
            VectorXi spans(dom_dim);                                       // current knot span in each dimension
            VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension  //p+1

            // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
            int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
            Bt.reserve(VectorXi::Constant(Bt.cols(), bf_per_pt)); //Bt.cols=2001, bf_per_pt=16 (nonzero values)

            vector<int> row_nnzs(Bt.rows(), 0); //Bt.rows(): # ofcontrol points

            // Iterate thru every point in subvolume given by tensor
            VectorX<T> param(input_BC.dom_dim);
            for (auto input_it = input_BC.begin(), input_end = input_BC.end(); input_it != input_end; ++input_it) //main loop, every row
            {

                input_it.params(param); //get parametric representation at the point,and  param is vector, iteration from the first point : input_it

                // Compute basis functions at input point
                for (auto k = 0; k < dom_dim; k++)
                {
                int p   = mfa_data.p(k); //polynomial degree =p
                T   u   = param(k);   //compute u and v
                spans[k] = mfa_data.tmesh.FindSpan(k, u); // compute not whole, but we find correspoinding(relavant) span to the point, because most of the basis functions for a specific point  are zero./ which basis functions are computed

                ctrl_starts(k) = spans[k] - p - t.knot_mins[k]; //indexing

                mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi); // compute nonzero basis function in 1D [k], => store results in lb[k]
                }

                // Compute matrix value and insert into Nt
                VolIterator ctrl_vol_iter(nctrl_patch); //get index for the patch that are relavant to the basis functions --> That is, the points in a small squre are indexed. nctrl_patch=p+1=4

                while (!ctrl_vol_iter.done())
                {
                int ctrl_idx_full = 0;
                T coeff_prod = 1;
                for (int k = 0; k < dom_dim; k++)
                {
                    int idx = ctrl_vol_iter.idx_dim(k);
                    ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k); //indexing
                    coeff_prod *= lB[k][idx];

                }

                // increment number of entries in this row (for use in constructing N)
                row_nnzs[ctrl_idx_full]++;

                Bt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;

                // FRAGILE command; reserve() must be called prior, **with vector signature**
                // entries must be inserted only once, and entries within a column must be inserted
                // in order of increasing row index.
                // This is an internal SparseMatrix command and no consistency checks are done.
                //
                // Empirical testing indicates that insertBackUncompressed is faster than insert, which
                // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
                // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

                ctrl_vol_iter.incr_iter();
                }
            }

            Bt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


            // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
            {
            // Must do this after Nt is in compressed mode!
            B.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

            int* col_starts     = B.outerIndexPtr();   //SparsMatrix outerIndexPtr, innerIndexPtr, valuePtr are inline functions, so the functions are used here, not re-call process (see inline functions in c++ tutorial)
            int* row_ids        = B.innerIndexPtr();
            T*   tr_values      = B.valuePtr();
            int* col_inner_nnzs = B.innerNonZeroPtr();

            int* oip    = Bt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
            int* iip    = Bt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
            T*   values = Bt.valuePtr();

            for (int o = 0; o < Bt.cols(); o++) // loop through columns of Nt (rows of N)
                {
                for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
                {
                int row = o;                // row in N
                int col = iip[oip[o]+i];    // col in N
                row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
                tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

                // col_nnzs tracks how many entries we have inserted into each col buffer
                // Since we loop through the rows of N in order, the row indices within
                //     each col buffer are strictly increasing
                col_inner_nnzs[col]++;
                }
	    }

	  B.makeCompressed();
	}




        fill_time = clock() - fill_time;
        if (verbose)
            cerr << "BC Matrix Assembly Total Time: " << setprecision(3) << ((double)fill_time)/CLOCKS_PER_SEC << "s." << endl;
      }

        // Assemble the unweighted analytic pure-partial derivative operator
        // used by the corrected Reg-MFA baseline.  The returned sparse matrix
        // is D_r^T: one row per control point and dom_dim*nctrl columns.  This
        // deliberately stops before the historical Yeh/local-strength path.
        void AssembleDerivativePenaltyTranspose(
                            TensorIdx               t_idx,
                            int                     derivative_order,
                            const VectorX<T>&       dom_mins,
                            const VectorX<T>&       dom_maxs,
                            SparseMatrixX<T>&       derivative_transpose)
        {
            if (derivative_order != 1 && derivative_order != 2)
                throw std::invalid_argument(
                    "Reg-MFA derivative order must be 1 or 2");

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
            const int num_ctrl = t.nctrl_pts.prod();
            const int num_derivative_rows = dom_dim * num_ctrl;
            const VectorX<T> extents = dom_maxs - dom_mins;

            if (derivative_transpose.rows() != num_ctrl ||
                derivative_transpose.cols() != num_derivative_rows)
                throw std::invalid_argument(
                    "Incorrect Reg-MFA derivative-transpose dimensions");
            for (int k = 0; k < dom_dim; ++k)
            {
                if (mfa_data.p(k) < derivative_order)
                    throw std::invalid_argument(
                        "Reg-MFA derivative order exceeds spline degree");
                if (!std::isfinite(static_cast<double>(extents(k))) ||
                    extents(k) <= T(0))
                    throw std::invalid_argument(
                        "Reg-MFA requires finite positive domain extents");
            }

            VectorXi spans(dom_dim);
            vector<MatrixX<T> > basis(dom_dim);
            for (int k = 0; k < dom_dim; ++k)
                basis[k].resize(
                    derivative_order + 1,
                    t.nctrl_pts(k));

            const int n_entries =
                (mfa_data.p.array() + 1).prod();
            derivative_transpose.reserve(
                VectorXi::Constant(
                    derivative_transpose.cols(),
                    n_entries));

            // Reuse the control-associated anchors of ConsMatrix(), including
            // its analytic basis-maximum search.
            vector<vector<T> > anchor_pts(dom_dim);
            for (int i = 0; i < dom_dim; ++i)
            {
                MatrixX<T> temp_ders(3, t.nctrl_pts(i));
                const int degree = mfa_data.p(i);
                anchor_pts[i].resize(t.nctrl_pts(i));
                anchor_pts[i][0] = T(0);
                anchor_pts[i][t.nctrl_pts(i) - 1] = T(1);
                for (int j = 1; j < t.nctrl_pts(i) - 1; ++j)
                {
                    T low = mfa_data.tmesh.all_knots[i][j];
                    T high =
                        mfa_data.tmesh.all_knots[i][j + degree + 1];
                    T mid = (low + high) / T(2);
                    temp_ders.setZero();
                    int span = mfa_data.tmesh.FindSpan(i, mid);
                    mfa_data.DerBasisFuns(
                        i, mid, span, 1, temp_ders);

                    while (temp_ders(1, j) < T(-0.01) ||
                           temp_ders(1, j) > T(0.01))
                    {
                        if (temp_ders(1, j) > T(0.01))
                            low = mid;
                        else
                            high = mid;
                        mid = (low + high) / T(2);
                        span = mfa_data.tmesh.FindSpan(i, mid);
                        mfa_data.DerBasisFuns(
                            i, mid, span, 1, temp_ders);
                    }
                    anchor_pts[i][j] = mid;
                }
            }

            VectorXi ctrl_starts(dom_dim);
            const VectorXi nctrl_patch =
                mfa_data.p + VectorXi::Ones(dom_dim);

            VolIterator point_iterator(t.nctrl_pts);
            while (!point_iterator.done())
            {
                for (int k = 0; k < dom_dim; ++k)
                {
                    const T parameter =
                        anchor_pts[k][point_iterator.idx_dim(k)];
                    const int degree = mfa_data.p(k);
                    spans[k] =
                        mfa_data.tmesh.FindSpan(k, parameter);
                    ctrl_starts(k) =
                        spans[k] - degree - t.knot_mins[k];
                    basis[k].setZero();
                    mfa_data.DerBasisFuns(
                        k,
                        parameter,
                        spans[k],
                        derivative_order,
                        basis[k]);
                }

                for (int derivative_dim = 0;
                     derivative_dim < dom_dim;
                     ++derivative_dim)
                {
                    VolIterator control_iterator(
                        nctrl_patch,
                        ctrl_starts,
                        t.nctrl_pts);
                    while (!control_iterator.done())
                    {
                        T derivative_value = T(1);
                        for (int k = 0; k < dom_dim; ++k)
                        {
                            const int basis_index =
                                control_iterator.idx_dim(k);
                            T factor = basis[k](
                                k == derivative_dim
                                    ? derivative_order
                                    : 0,
                                basis_index);
                            if (k == derivative_dim)
                                factor /= std::pow(
                                    extents(k),
                                    derivative_order);
                            derivative_value *= factor;
                        }

                        derivative_transpose.insertBackUncompressed(
                            control_iterator.cur_iter_full(),
                            point_iterator.cur_iter() * dom_dim +
                                derivative_dim) = derivative_value;
                        control_iterator.incr_iter();
                    }
                }
                point_iterator.incr_iter();
            }

            derivative_transpose.makeCompressed();
            if (derivative_transpose.rows() != num_ctrl ||
                derivative_transpose.cols() != num_derivative_rows)
                throw std::runtime_error(
                    "Reg-MFA derivative operator changed dimensions");
            for (int i = 0;
                 i < derivative_transpose.nonZeros();
                 ++i)
                if (!std::isfinite(
                        static_cast<double>(
                            derivative_transpose.valuePtr()[i])))
                    throw std::runtime_error(
                        "Reg-MFA derivative operator contains "
                        "a non-finite entry");
        }

        void ConsMatrix(    TensorIdx               t_idx,
                            int                     deriv,
                            T                       c_target,
                            const SparseMatrixX<T>& N,
                            const SparseMatrixX<T>& Nt,
                            const VectorX<T>&       dom_mins,
                            const VectorX<T>&       dom_maxs,
                            SparseMatrixX<T>&       Ct)
        {
            clock_t fill_time = clock();
                cerr << "Adjusting matrix for regularization..." << endl;

            const int num_ctrl = N.cols();
            const int num_cons = dom_dim * num_ctrl;
            const VectorX<T> extents = dom_maxs - dom_mins;

            if (Ct.rows() != num_ctrl || Ct.cols() != num_cons)
            {
                fmt::print("ERROR: Incorrect matrix dimensions of Ct in ConsMatrix()\nExiting.\n");
                exit(1);
            }

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
            VectorXi spans(dom_dim);

            vector<MatrixX<T>> B(dom_dim);
            for (int k = 0; k < dom_dim; k++)
            {
                B[k].resize(deriv+1, t.nctrl_pts(k));
            }

            int n_entries = (mfa_data.p.array() + 1).prod();
            Ct.reserve(VectorXi::Constant(Ct.cols(), n_entries));

            // Compute parameters for each derivative constraint
            // Compute basis functions and derivatives at each parameter
            vector<vector<T>> anchor_pts(dom_dim);
            for (int i = 0; i < dom_dim; i++)
            {
                MatrixX<T> temp_ders(3, t.nctrl_pts(i)); // 3 rows stores up to 2nd deriv

                int pc = mfa_data.p(i);     // current p for this dimension

                anchor_pts[i].resize(t.nctrl_pts(i));
                anchor_pts[i][0] = 0;
                anchor_pts[i][t.nctrl_pts(i)-1] = 1.0;
                for (int j = 1; j < t.nctrl_pts(i) - 1; j++)
                {
                    T low = mfa_data.tmesh.all_knots[i][j];
                    T high = mfa_data.tmesh.all_knots[i][j+pc+1];
                    T mid = (low + high) / 2;
                    int temp_span = 0;

                    // Find parameter for max value of basis function with binary search
                    temp_ders.setZero();
                    temp_span = mfa_data.tmesh.FindSpan(i, mid);
                    mfa_data.DerBasisFuns(i, mid, temp_span, 1, temp_ders);

                    // TODO speed up by using Newton's method instead.
                    //      Warning: Need to ensure convergence to the right vanishing derivative when using Newton's method (there are three per basis fxn)
                    while (temp_ders(1,j) < -0.01 || temp_ders(1,j) > 0.01)
                    {
                        if (temp_ders(1,j) > 0.01)    // if function is increasing, we are left of max value
                            low = mid;
                        else
                            high = mid;
                        mid = (low + high) / 2;

                        temp_span = mfa_data.tmesh.FindSpan(i, mid);
                        mfa_data.DerBasisFuns(i, mid, temp_span, 1, temp_ders);
                    }

                    anchor_pts[i][j] = mid;
                }
            }

            VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
            VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

            // Loop through each parameter corresponding to derivative constraint
            VolIterator pt_it(t.nctrl_pts);
            while (!pt_it.done())
            {
                // Compute basis function values and derivatives at each anchor
                for (int k = 0; k < dom_dim; k++)
                {
                    T u = anchor_pts[k][pt_it.idx_dim(k)];
                    int p = mfa_data.p(k);
                    spans[k] = mfa_data.tmesh.FindSpan(k, u);
                    ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

                    B[k].setZero();
                    mfa_data.DerBasisFuns(k, u, spans[k], deriv, B[k]);
                }

                // Compute constraints for each directional derivative
                for (int dk = 0; dk < dom_dim; dk++)
                {
                    VolIterator ctrl_vol_iter(nctrl_patch, ctrl_starts, t.nctrl_pts);
                    while (!ctrl_vol_iter.done())   // for each nonzero basis function at the location for the constraint
                    {
                        int ctrl_idx_full = ctrl_vol_iter.cur_iter_full();

                        T current_deriv = 1;
                        for (auto k = 0; k < dom_dim; k++) // compute tensor-product basis function
                        {
                            int idx = ctrl_vol_iter.idx_dim(k);

                            T mult = B[k]((k==dk) ? deriv : 0, idx);    // deriv if k==dk, otherwise normal basis function

                            if (k == dk)    // if this is a derivative, scale properly
                                mult /= pow(extents(k), deriv);

                            current_deriv *= mult;
                        }

                        Ct.insertBackUncompressed(ctrl_idx_full, pt_it.cur_iter() * dom_dim + dk) = current_deriv;

                        ctrl_vol_iter.incr_iter();
                    }
                }
                pt_it.incr_iter();
            }

            VectorX<T> reg_strengths(num_ctrl);

            yeh_reg(t, N, Nt, Ct, reg_strengths, deriv, c_target);

            write_reg_strength(t, deriv, reg_strengths, anchor_pts);

            fill_time = clock() - fill_time;
                cerr << "Regularization Total Time: " << setprecision(3) << ((double)fill_time)/CLOCKS_PER_SEC << "s." << endl;
        }

        void uniform_reg(TensorProduct<T>& t,
                            const SparseMatrixX<T>& N,
                            const SparseMatrixX<T>& Nt,
                            SparseMatrixX<T>& Ct,
                            VectorX<T>& reg_strengths,
                            int deriv,
                            T c_target)
        {
            reg_strengths = VectorX<T>::Ones(t.nctrl_pts.prod());
            reg_strengths = c_target*reg_strengths;

            Ct = c_target*Ct;

            return;
        }

        void yeh_reg(TensorProduct<T>& t,
                            const SparseMatrixX<T>& N,
                            const SparseMatrixX<T>& Nt,
                            SparseMatrixX<T>& Ct,
                            VectorX<T>& reg_strengths,
                            int deriv,
                            T c_target)
        {
            // Compute regularization strengths
            SparseMatrixX<T> C(Ct.transpose());                             // create col-major transpose for fast column sums
            Eigen::DiagonalMatrix<T, Eigen::Dynamic> lambda(C.rows());       // NEW test case
            T str = 0;
            for (int i = 0; i < C.cols(); i++)
            {
                T n_sum = N.col(i).sum();
                T c_sum = C.col(i).cwiseAbs().sum();
                if (deriv == 1 && n_sum > 0)
                    str = 0;
                else
                    str = max((c_target - n_sum)/c_sum, (T)0);

                reg_strengths(i) = str; // nb. for debug below only

                for (int k = 0; k < dom_dim; k++)
                {
                    lambda.diagonal()(dom_dim*i + k) = str;
                }
            }

            // Scale the constraint matrix Ct by the regularization strengths
            // Each col of Ct (row of C) is multiplied by a diagonal entry of lambda
            Ct = Ct*lambda;
            return;
        }

        void vazquez_reg(TensorProduct<T>& t,
                            const SparseMatrixX<T>& N,
                            const SparseMatrixX<T>& Nt,
                            SparseMatrixX<T>& Ct,
                            VectorX<T>& reg_strengths,
                            int deriv,
                            T c_target) // should always be =2 for vazquez's method
        {
            if (deriv != 2)
            {
                cerr << "ERROR: deriv order not equal to 2 in Encoder::vazquez_reg()\nExiting" << endl;
                exit(0);
            }

            T str = 0;
            int lid = 0;
            VectorXi ijk(dom_dim);
            VectorXi ijk_r(dom_dim);
            GridInfo g;
            g.init(dom_dim, t.nctrl_pts);

            SparseMatrixX<T> NtN = Nt * Nt.transpose();

            Eigen::DiagonalMatrix<T, Eigen::Dynamic> lambda(Ct.cols());

            for (int j = 0; j < t.nctrl_pts.prod(); j++) // loop thru cols of NtN
            {
                T sum0 = 0, sum1 = 0;
                g.idx2ijk(j, ijk_r); // ijk coords of basis functions for col j

                VolIterator pt_it(t.nctrl_pts);
                while (!pt_it.done())
                {
                    lid = pt_it.cur_iter();
                    ijk = pt_it.idx_dim();


                    if ((ijk_r-ijk)(0) % 2 == 1) // if x offset between basis funs is odd
                    {
                        sum0 -= NtN.coeff(lid, j);
                    }
                    else
                    {
                        sum0 += NtN.coeff(lid, j);
                    }

                    if ((ijk_r-ijk)(1) % 2 == 1) // if y offset between basis funs is odd
                    {
                        sum1 -= NtN.coeff(lid, j);
                    }
                    else
                    {
                        sum1 += NtN.coeff(lid, j);
                    }

                    pt_it.incr_iter();
                }
                str = c_target * max(0.0, max(.11111 - abs(sum0), .11111 - abs(sum1)));
                reg_strengths(j) = str;
                for (int k = 0; k < dom_dim; k++)
                {
                    lambda.diagonal()(dom_dim*j + k) = str;
                }
            }

            Ct = Ct*lambda;     // right multiply for vazquez method

            return;
        }

        void write_reg_strength(TensorProduct<T>& t,
                                int deriv,
                                VectorX<T>& reg_strengths,
                                vector<vector<T>>& anchor_pts)
        {
            // Export the regularization strength and position for each term.
            ofstream reg_st_out;
            string reg_st_fname = "reg-strength-" + to_string(deriv) + ".txt";
            reg_st_out.open(reg_st_fname);

            VolIterator pt_it(t.nctrl_pts);
            while (!pt_it.done())
            {
                for (int i = 0; i < dom_dim; i++)
                {
                    int id = pt_it.idx_dim(i);
                    reg_st_out << anchor_pts[i][id] << " ";
                }
                reg_st_out << reg_strengths(pt_it.cur_iter()) << endl;

                pt_it.incr_iter();
            }
            reg_st_out.close();
        }

        // Assemble B-spline collocation matrix for a full tensor, using sparse matrices
        // Here we are filling Nt (transpose of N)
        // Nt is a matrix of n x m scalars that are the basis function coefficients
        //  _                                      _
        // |  N_0(u[0])     ...     N_0(u[m-1])     |
        // |     ...        ...      ...            |
        // |  N_n-1(u[0])   ...     N_n-1(u[m-1])   |
        //  -                                      -
        void CollMatrixUnified( TensorIdx           t_idx,    // index of tensor product containing input points and control points
                                SparseMatrixX<T>&   N, // (output) collocation matrix
                                SparseMatrixX<T>&   Nt)  // (output) transpose of collocation matrix
        {
            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
            assert(Nt.rows() == t.nctrl_pts.prod());
            assert(Nt.cols() == input.npts);

            clock_t fill_time = clock();

            // resize matrices in case number of control points changed
            const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());

            // control point strides
            VectorXi cps = VectorXi::Ones(dom_dim);
            for (int k = 1; k < dom_dim; k++)
            {
                cps(k) = cps(k-1) * t.nctrl_pts(k-1);
            }

            // Prep for evaluating basis functions
            VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
            BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
            vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
            for (int k = 0; k < dom_dim; k++)
            {
                lB[k].resize(q[k]);
            }

            VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
            VectorXi spans(dom_dim);                                       // current knot span in each dimension
            VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

            // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
            int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
            Nt.reserve(VectorXi::Constant(Nt.cols(), bf_per_pt));

            vector<int> row_nnzs(Nt.rows(), 0);

            // Iterate thru every point in subvolume given by tensor
            VectorX<T> param(input.dom_dim);
            for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
            {
                input_it.params(param);

                // Compute basis functions at input point
                for (auto k = 0; k < dom_dim; k++)
                {
                    int p   = mfa_data.p(k);
                    T   u   = param(k);

                    spans[k] = mfa_data.tmesh.FindSpan(k, u);

                    ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

                    mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi);
                }

                // Compute matrix value and insert into Nt
                VolIterator ctrl_vol_iter(nctrl_patch);
                while (!ctrl_vol_iter.done())
                {
                    int ctrl_idx_full = 0;
                    T coeff_prod = 1;
                    for (int k = 0; k < dom_dim; k++)
                    {
                        int idx = ctrl_vol_iter.idx_dim(k);
                        ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
                        coeff_prod *= lB[k][idx];
                    }

                    // increment number of entries in this row (for use in constructing N)
                    row_nnzs[ctrl_idx_full]++;

                    Nt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;
                    // FRAGILE command; reserve() must be called prior, **with vector signature**
                    // entries must be inserted only once, and entries within a column must be inserted
                    // in order of increasing row index.
                    // This is an internal SparseMatrix command and no consistency checks are done.
                    //
                    // Empirical testing indicates that insertBackUncompressed is faster than insert, which
                    // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
                    // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

                    ctrl_vol_iter.incr_iter();
                }
            }

            Nt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


            // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
            {
                // Must do this after Nt is in compressed mode!
                N.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

                int* col_starts     = N.outerIndexPtr();
                int* row_ids        = N.innerIndexPtr();
                T*   tr_values      = N.valuePtr();
                int* col_inner_nnzs = N.innerNonZeroPtr();

                int* oip    = Nt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
                int* iip    = Nt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
                T*   values = Nt.valuePtr();

                for (int o = 0; o < Nt.cols(); o++) // loop through columns of Nt (rows of N)
                {
                    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
                    {
                        int row = o;                // row in N
                        int col = iip[oip[o]+i];    // col in N
                        row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
                        tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

                        // col_nnzs tracks how many entries we have inserted into each col buffer
                        // Since we loop through the rows of N in order, the row indices within
                        //     each col buffer are strictly increasing
                        col_inner_nnzs[col]++;
                    }
                }

                N.makeCompressed();
            }

            fill_time = clock() - fill_time;
        }

    void CollMatrixUnifiedDers( TensorIdx           t_idx,    // index of tensor product containing input points and control points
				VectorXi derivs, // derv with respect to which dom_min
                                SparseMatrixX<T>&   N, // (output) collocation matrix
                                SparseMatrixX<T>&   Nt)  // (output) transpose of collocation matrix
    {
      TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
      assert(Nt.rows() == t.nctrl_pts.prod());
      assert(Nt.cols() == input.npts);

      clock_t fill_time = clock();

      // resize matrices in case number of control points changed
      const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
      t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
      t.weights.resize(t.ctrl_pts.rows());

      // control point strides
      VectorXi cps = VectorXi::Ones(dom_dim);
      for (int k = 1; k < dom_dim; k++)
	{
	  cps(k) = cps(k-1) * t.nctrl_pts(k-1);
	}

      // Prep for evaluating basis functions
      VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
      BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
      //For PDE-constrained, in case of dom_dim=2
      vector<vector<T>>  D_dim0(3);      //dom_dim=0
      vector<vector<T>>  D_dim1(3);      //dom_dim=1
      vector<vector<T>>  D_dim2(3);      //dom_dim=2
      for (int nd = 0; nd < 3; nd++)
	{
	  D_dim0[nd].resize(q[0]);
	  D_dim1[nd].resize(q[1]);
	  D_dim2[nd].resize(q[2]);
	  }

      VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
      VectorXi spans(dom_dim);                                       // current knot span in each dimension
      VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

      // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
      int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
      Nt.reserve(VectorXi::Constant(Nt.cols(), bf_per_pt));

      vector<int> row_nnzs(Nt.rows(), 0);

      // Iterate thru every point in subvolume given by tensor
      VectorX<T> param(input.dom_dim);
      for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
	{
	  input_it.params(param);

	  // Compute basis functions at input point
	  for (auto k = 0; k < dom_dim; k++)
	    {
	      int p   = mfa_data.p(k);
	      T   u   = param(k);

	      spans[k] = mfa_data.tmesh.FindSpan(k, u);

	      ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

	      //Nx=BxB ,Nt=BBt,Nxx=BxxB
		int derivv=derivs(k);
		if (k==0){
	      std::cout << "derivs("<<k<<"): " << derivs(k) << "testing"<<std::endl;
		  mfa_data.FastBasisFunsDers(k, u, spans[k],derivv, D_dim0, bfi);
		}
		  else if (k==1){
	      std::cout << "derivs("<<k<<"): " << derivs(k) << "testing"<<std::endl;
		    mfa_data.FastBasisFunsDers(k, u, spans[k],derivv, D_dim1, bfi);
		  }
		  else if (k==2){
	      std::cout << "derivs("<<k<<"): " << derivs(k) << "testing"<<std::endl;
		    mfa_data.FastBasisFunsDers(k, u, spans[k],derivv, D_dim2, bfi);
		  }

	    }
	  // Compute matrix value and insert into Nt
		  VolIterator ctrl_vol_iter(nctrl_patch);
		  while (!ctrl_vol_iter.done())
		    {
		      int ctrl_idx_full = 0;
		      T coeff_prod = 1;
		      for (int k = 0; k < dom_dim; k++)
			{
			  int idx = ctrl_vol_iter.idx_dim(k);
			  ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
			  int nderivs=derivs(k);
			  if (k==0){
			    coeff_prod *= D_dim0[nderivs][idx];
			  }
			    else if (k==1){
			      coeff_prod *= D_dim1[nderivs][idx];
			    }
			    else if (k==2){
			      coeff_prod *= D_dim2[nderivs][idx];
			    }
			  }

	      // increment number of entries in this row (for use in constructing N)
	      row_nnzs[ctrl_idx_full]++;

	      Nt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;
	      // FRAGILE command; reserve() must be called prior, **with vector signature**
	      // entries must be inserted only once, and entries within a column must be inserted
	      // in order of increasing row index.
	      // This is an internal SparseMatrix command and no consistency checks are done.
	      //
	      // Empirical testing indicates that insertBackUncompressed is faster than insert, which
	      // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
	      // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

	      ctrl_vol_iter.incr_iter();
	    }
	}

      Nt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form


      // Fill the matrix directly as the transpose to avoid a temporary sparse transpose.
      {
	// Must do this after Nt is in compressed mode!
	N.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

	int* col_starts     = N.outerIndexPtr();
	int* row_ids        = N.innerIndexPtr();
	T*   tr_values      = N.valuePtr();
	int* col_inner_nnzs = N.innerNonZeroPtr();

	int* oip    = Nt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
	int* iip    = Nt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
	T*   values = Nt.valuePtr();

	for (int o = 0; o < Nt.cols(); o++) // loop through columns of Nt (rows of N)
	  {
	    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
	      {
		int row = o;                // row in N
		int col = iip[oip[o]+i];    // col in N
		row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
		tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

		// col_nnzs tracks how many entries we have inserted into each col buffer
		// Since we loop through the rows of N in order, the row indices within
		//     each col buffer are strictly increasing
		col_inner_nnzs[col]++;
	      }
	  }

	N.makeCompressed();
      }

      fill_time = clock() - fill_time;
    }

#ifdef MFA_TBB
        // Computes the product of two Eigen sparse matrices using TBB.  All matrices must be in column major format.
        void MatProdThreaded(   Eigen::SparseMatrix<T, Eigen::ColMajor>& lhs,
                                Eigen::SparseMatrix<T, Eigen::ColMajor>& rhs,
                                Eigen::SparseMatrix<T, Eigen::ColMajor>& res,
                                int reserveSizes )
        {
            using LhsInIt = typename Eigen::SparseMatrix<T, Eigen::ColMajor>::InnerIterator;
            using RhsInIt = typename Eigen::SparseMatrix<T, Eigen::ColMajor>::InnerIterator;

            int rows = lhs.innerSize();
            int cols = rhs.outerSize();

            res.setZero();
            res.reserve(VectorXi::Constant(rows, reserveSizes));

            // Internal SparseMatrix buffers
            int* outer_index = res.outerIndexPtr();
            int* inner_nnz = res.innerNonZeroPtr();
            auto& res_data = res.data();

            tbb::affinity_partitioner ap;
            parallel_for(blocked_range<size_t>(0, cols), [&](blocked_range<size_t>& r)
            {
                int j_begin = r.begin();
                int j_end = r.end();

                std::vector<char> mask(rows, 0);
                std::vector<T> values(rows, 0);

                // we compute each column of the result, one after the other
                for (int j=j_begin; j<j_end; ++j)
                {
                    for (RhsInIt rhsIt(rhs, j); rhsIt; ++rhsIt)
                    {
                        T y = rhsIt.value();
                        int k = rhsIt.index();
                        for (LhsInIt lhsIt(lhs, k); lhsIt; ++lhsIt)
                        {
                            int i = lhsIt.index();
                            T x = lhsIt.value();
                            if(mask[i] == 0)
                            {
                                mask[i] = 1;
                                values[i] = x * y;
                            }
                            else
                                values[i] += x * y;
                        }
                    }

                    // TODO: can loop over nonzeros only, but this involves sorted insertion, which may be slower
                    for(int i=0; i<rows; ++i)
                    {
                        if(mask[i] == 1)
                        {
                            mask[i] = 0;

                            // We never want this (slow) reallocation to happen. Always OVER-estimate reserveSizes (input variable)
                            if( inner_nnz[j] >= outer_index[j+1] - outer_index[j] )
                            {
                                cerr << "Warning: Reallocating memory in threaded matrix product. Did you reserve space properly?" << endl;
                                VectorXi extra_size = VectorXi::Zero(cols);
                                extra_size[j] = std::max<int>(2, inner_nnz[j]);
                                res.reserve(extra_size);  // NB: reserve() always adds to existing buffer
                            }

                            // Insert entry
                            int p = outer_index[j] + inner_nnz[j];
                            res_data.index(p) = i;
                            res_data.value(p) = values[i];
                            inner_nnz[j]++;
                        }
                    }
                }

            }, ap );

            res.makeCompressed();
        }

#endif // MFA_TBB

        // Encodes ctrl points in each dimensions simultaneously
        // Necessary for encoding unstructured input data, where parameter values
        // corresponding to input do not lie on structured grid
        //
        // TODO: Currently only valid for one tensor product. To extend to general
        // tmesh, need a way to compute the total number of input points in a given
        // tensor. At the moment, this is only possible with the domain_pts()
        // method, which does not work for unstructured data
        void EncodeUnified( TensorIdx   t_idx,                      // tensor product being encoded
                            T           regularization=0,           // parameter to set smoothing of artifacts from non-uniform data
                            bool        reg1and2=false,
                            bool        weighted=true,   // solve for and use weights
			                bool        BC=true)        //constaint on Boundary condition
        {
            {
                fmt::print("Encoding dimensions all at once\n");
            }
            if (verbose >= 3)
            {
                fmt::print("  => NOTE: Unified encode only valid for single tensor product!\n");
            }

            if (weighted)  // We want weighted encoding to be default behavior eventually. However, not currently implemented.
            {
                fmt::print("ERROR: NURBS (nonuniform weights) are not implemented for unified-dimensional encoding!\n");
                exit(1);
            }

            const int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                           // control point dimensonality
            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
            SparseMatrixX<T> N(input.npts, t.nctrl_pts.prod());
            SparseMatrixX<T> Nt(t.nctrl_pts.prod(), input.npts);
            CollMatrixUnified(t_idx, N, Nt);

            // Construct RHS of linear system
            MatrixX<T>  R(Nt.cols(), pt_dim);

            RHSUnified(Nt, R);

	    int bcline = 398;
	    int num_reg_conds = t.nctrl_pts.prod();

	      SparseMatrixX<T> B(bcline, num_reg_conds);
	      SparseMatrixX<T> Bt(num_reg_conds,bcline);
	      SparseMatrixX<T> Bt1(num_reg_conds,bcline);


            // Construct RHS of linear system

	    regularization=0;
            if (regularization > 0)
            {
                    fmt::print("Applying model regularization with strength r={}\n", regularization);

                SparseMatrixX<T> Ct(Nt.rows(), num_reg_conds * dom_dim);
                SparseMatrixX<T> Ct1(Nt.rows(), num_reg_conds * dom_dim);
                SparseMatrixX<T> Ct2(Nt.rows(), num_reg_conds * dom_dim);
		reg1and2=0;
                if (reg1and2)   // constrain 1st and 2nd derivs
                {
                    ConsMatrix(t_idx, 1, regularization, N, Nt, input.mins(), input.maxs(), Ct1);
                    ConsMatrix(t_idx, 2, regularization, N, Nt, input.mins(), input.maxs(), Ct2);

                    Ct.conservativeResize(Ct1.rows(), Ct1.cols() + Ct2.cols());
                    Ct.leftCols(Ct1.cols()) = Ct1;
                    Ct.rightCols(Ct2.cols()) = Ct2;
                    Ct.makeCompressed();
                }
                else            // constrain only 2nd derivs
                {
                    ConsMatrix(t_idx, 2, regularization, N, Nt, input.mins(), input.maxs(), Ct2);
                    Ct = Ct2;            // only C2 regularization
                }

                // Concatenate Ct to the right end of Nt
                Nt.conservativeResize(Nt.rows(), Nt.cols() + Ct.cols());
                Nt.rightCols(Ct.cols()) = Ct;
                Nt.makeCompressed();
            }
	    BC=false;
	    if (BC>0){
	      BCMatrix(t_idx,B,Bt1);
	      Bt = Bt1;            // only C2 regularization

	      Nt.conservativeResize(Nt.rows(), Nt.cols() + Bt.cols()); //adding the columns number to the Nt matrix
	      Nt.rightCols(Bt.cols()) = Bt; //adding the columns to the Nt matrix
	      Nt.makeCompressed(); //making sparse matrix
	    }
		// Set up linear system
	SparseMatrixX<T> Mat(Nt.rows(), Nt.rows()); // Mat will be the matrix on the LHS

	#ifdef MFA_TBB  // TBB version
	Eigen::SparseMatrix<T, Eigen::ColMajor> NCol = Nt.transpose();

	int ntn_sparsity = (2*mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
	MatProdThreaded(Nt, NCol, Mat, ntn_sparsity);
	#else
	Mat = Nt * Nt.transpose();
	#endif

            // Solve Linear System
            Eigen::ConjugateGradient<SparseMatrixX<T>, Eigen::Lower|Eigen::Upper>  solver;  // Default preconditioner is Jacobi

            solver.setTolerance(1e-8);

            solver.compute(Mat);
            if (solver.info() != Eigen::Success)
                fmt::print("WARNING: Matrix decomposition failed in EncodeTensor\n");
            else if (verbose >= 2)
                fmt::print("Sparse matrix factorization successful\n");

            t.ctrl_pts = solver.solve(R);
	    static int step = 0; // `static` keeps its value between loop iterations
	    // Inside the optimization loop, every time P is updated:
	    std::ostringstream filename;
	    filename << "P_vector_step_" << std::setw(4) << std::setfill('0') << step << ".txt";
	    std::ofstream p_file(filename.str(), std::ios::out);

	    if (p_file.is_open())
	      {
		p_file << t.ctrl_pts<< std::endl;
		p_file.close();
	      }
	    else
	      {
		std::cerr << "Failed to open file: " << filename.str() << std::endl;
	      }

	    ++step; // Increment step at end of iteration

            if (solver.info() != Eigen::Success)
            {
                fmt::print("WARNING: Least-squares solve failed in EncodeTensor\n");
                fmt::print("  error: {} (tolerance = {})\n", solver.error(), solver.tolerance());
                fmt::print("  # iterations: {} (max iterations = {})\n", solver.iterations(), solver.maxIterations());
            }
            else if (verbose >= 2)
            {
                fmt::print("Sparse matrix solve successful\n");
                fmt::print("  # iterations: {}\n", solver.iterations());
            }
        }
      //Physics-MFA

      void EncodeUnifiedBFGS( TensorIdx   t_idx,                      // tensor product being encoded
			      T           regularization=0,           // parameter to set smoothing of artifacts from non-uniform data
			      bool        reg1and2=false,
			      bool        weighted=true,   // solve for and use weights
			      bool        BC=true)        //constaint on Boundary condition
      {
	  {
	    fmt::print("Encoding dimensions all at once\n");
	  }
	if (verbose >= 3)
	  {
	    fmt::print("  => NOTE: Unified encode only valid for single tensor product!\n");
	  }

	if (weighted)  // We want weighted encoding to be default behavior eventually. However, not currently implemented.
	  {
	    fmt::print("ERROR: NURBS (nonuniform weights) are not implemented for unified-dimensional encoding!\n");
	    exit(1);
	  }

	const int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                           // control point dimensonality
	TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
	TensorProduct<T>& t2 = mfa_data2.tmesh.tensor_prods[t_idx];
	TensorProduct<T>& t3 = mfa_data3.tmesh.tensor_prods[t_idx];
	const RegMFAConfig baseline_config = ReadRegMFAConfig();
	const NSTraceAwareLSConfig trace_aware_ls_config =
	  ReadNSTraceAwareLSConfig();
	ValidateRegMFAConfig(
	    baseline_config,
	    static_cast<double>(regularization),
	    reg1and2);
	if (baseline_config.baseline_requested)
	  {
	    if (dom_dim != 3)
	      throw std::runtime_error(
	        "The NS Reg-MFA baseline contract requires x,y,t");
	    if (pt_dim != 1)
	      throw std::runtime_error(
	        "The NS Reg-MFA baseline requires scalar u, v, and p fields");
	    if (t2.nctrl_pts != t.nctrl_pts ||
	        t3.nctrl_pts != t.nctrl_pts)
	      throw std::runtime_error(
	        "The NS Reg-MFA baseline requires identical u/v/p lattices");
	    fmt::print(
	      "PIMFA_BASELINE_CONFIG requested_kind={} effective_kind={} "
	      "requested_alpha1={:.17g} effective_alpha1={:.17g} "
	      "requested_alpha2={:.17g} effective_alpha2={:.17g} "
	      "initializer_only={} role_sha256={} code_sha256={}\n",
	      BaselineKindName(baseline_config.kind),
	      BaselineKindName(baseline_config.kind),
	      baseline_config.alpha1,
	      baseline_config.alpha1,
	      baseline_config.alpha2,
	      baseline_config.alpha2,
	      baseline_config.initializer_only ? 1 : 0,
	      baseline_config.role_sha256,
	      baseline_config.code_sha256);
	  }

	const int num_reg_conds = t.nctrl_pts.prod();
	std::string pressure_init_mode("auto");
	if (const char* value = std::getenv("MFA_INIT_PRESSURE_MODE"))
	  pressure_init_mode = value;
	const T pressure_data_weight =
	  readEnv<T>("MFA_LAMBDA_DATA_P", T(1.0));
	bool pressure_init_uses_data = false;
	if (pressure_init_mode == "auto")
	  pressure_init_uses_data = pressure_data_weight > T(0);
	else if (pressure_init_mode == "data" ||
	         pressure_init_mode == "full-data")
	  pressure_init_uses_data = true;
	else if (pressure_init_mode == "zero")
	  pressure_init_uses_data = false;
	else
	  throw std::invalid_argument(
	    "MFA_INIT_PRESSURE_MODE must be one of: auto, data, "
	    "full-data, zero");
	const std::string pressure_checkpoint_regime =
	  pressure_init_uses_data ? "data" : "zero";
	if (baseline_config.baseline_requested && pressure_init_uses_data)
	  throw std::invalid_argument(
	    "Corrected NS baseline requires the withheld-pressure "
	    "initializer regime (set MFA_INIT_PRESSURE_MODE=zero and "
	    "MFA_LAMBDA_DATA_P=0)");

	// In Stage B, a generic initializer checkpoint is the complete source of
	// the starting coefficients.  Validate it before reading any field values
	// for the legacy Stage-A right-hand side.  A bad checkpoint must terminate
	// here instead of silently falling back to the ill-conditioned data solve.
	const NSInitializerCheckpointPlan initializer_checkpoint_plan =
	  ReadNSInitializerCheckpointPlan();
	if (initializer_checkpoint_plan.bypass_data_solve &&
	    baseline_config.baseline_requested)
	  throw std::invalid_argument(
	    "MFA_INIT_CHECKPOINT_IN cannot be combined with corrected "
	    "Reg-MFA baseline mode");
	const double initializer_coefficient_bound = 100.0;
	std::vector<double> initializer_checkpoint_payload;
	if (initializer_checkpoint_plan.bypass_data_solve)
	  initializer_checkpoint_payload =
	    ReadNSInitializerCheckpointV1Strict(
	      initializer_checkpoint_plan.path,
	      pressure_checkpoint_regime,
	      static_cast<size_t>(3) * static_cast<size_t>(num_reg_conds),
	      initializer_coefficient_bound);

	SparseMatrixX<T> N(input.npts, t.nctrl_pts.prod());
	SparseMatrixX<T> Nt(t.nctrl_pts.prod(), input.npts);
	CollMatrixUnified(t_idx, N, Nt);
	const size_t n_train = ApplyInitializerFitMask(Nt, "NS");
	ValidateNSTraceAwareLSContext(
	  trace_aware_ls_config,
	  n_train,
	  static_cast<size_t>(num_reg_conds),
	  initializer_checkpoint_plan.bypass_data_solve,
	  baseline_config.baseline_requested,
	  pressure_init_uses_data);
	SparseMatrixX<T> Nt_data;
	if (baseline_config.baseline_requested)
	  Nt_data = Nt;

	// Construct RHS of linear system
	MatrixX<T>  R(Nt.cols(), pt_dim);
	MatrixX<T>  R2(Nt.cols(), pt_dim);
	MatrixX<T>  R3(Nt.cols(), pt_dim);
	if (!initializer_checkpoint_plan.bypass_data_solve)
	  {
	    RHSUnified(Nt, R);
	    RHSUnified2(Nt, R2);
	    RHSUnified3(Nt, R3);
	  }

		SparseMatrixX<T> derivative1_transpose;
	SparseMatrixX<T> derivative2_transpose;
	T derivative1_scale = T(0);
	T derivative2_scale = T(0);

	if (baseline_config.baseline_requested)
	  {
	    const size_t derivative_rows =
	      static_cast<size_t>(dom_dim) *
	      static_cast<size_t>(num_reg_conds);

	    if (baseline_config.alpha1 > 0.0)
	      {
		derivative1_transpose.resize(
		  Nt.rows(),
		  num_reg_conds * dom_dim);
		AssembleDerivativePenaltyTranspose(
		  t_idx,
		  1,
		  input.mins(),
		  input.maxs(),
		  derivative1_transpose);
		derivative1_scale = static_cast<T>(
		  TikhonovColumnScale(
		    baseline_config.alpha1,
		    n_train,
		    derivative_rows));
		SparseMatrixX<T> scaled = derivative1_transpose;
		scaled *= derivative1_scale;
		const int old_columns = Nt.cols();
		Nt.conservativeResize(
		  Nt.rows(),
		  old_columns + scaled.cols());
		Nt.rightCols(scaled.cols()) = scaled;
		Nt.makeCompressed();
	      }

	    if (baseline_config.alpha2 > 0.0)
	      {
		derivative2_transpose.resize(
		  Nt.rows(),
		  num_reg_conds * dom_dim);
		AssembleDerivativePenaltyTranspose(
		  t_idx,
		  2,
		  input.mins(),
		  input.maxs(),
		  derivative2_transpose);
		derivative2_scale = static_cast<T>(
		  TikhonovColumnScale(
		    baseline_config.alpha2,
		    n_train,
		    derivative_rows));
		SparseMatrixX<T> scaled = derivative2_transpose;
		scaled *= derivative2_scale;
		const int old_columns = Nt.cols();
		Nt.conservativeResize(
		  Nt.rows(),
		  old_columns + scaled.cols());
		Nt.rightCols(scaled.cols()) = scaled;
		Nt.makeCompressed();
	      }
	  }
	else
	  {
	    if (regularization != T(0))
	      std::cerr
		<< "Ignoring regularization=" << regularization
		<< " in the reusable data-only NS initializer; using 0 instead.\n";
	    regularization=0;
	    if (regularization > 0)
	      {
		  fmt::print("Applying model regularization with strength r={}\n", regularization);

                SparseMatrixX<T> Ct(Nt.rows(), num_reg_conds * dom_dim);
                SparseMatrixX<T> Ct1(Nt.rows(), num_reg_conds * dom_dim);
                SparseMatrixX<T> Ct2(Nt.rows(), num_reg_conds * dom_dim);
		reg1and2=1;
                if (reg1and2)   // constrain 1st and 2nd derivs
		  {

                    ConsMatrix(t_idx, 1, regularization, N, Nt, input.mins(), input.maxs(), Ct1);
                    ConsMatrix(t_idx, 2, regularization, N, Nt, input.mins(), input.maxs(), Ct2);

                    Ct.conservativeResize(Ct1.rows(), Ct1.cols() + Ct2.cols());
                    Ct.leftCols(Ct1.cols()) = Ct1;
                    Ct.rightCols(Ct2.cols()) = Ct2;
                    Ct.makeCompressed();
		  }
                else            // constrain only 2nd derivs
		  {
                    ConsMatrix(t_idx, 2, regularization, N, Nt, input.mins(), input.maxs(), Ct2);
                    Ct = Ct2;            // only C2 regularization
		  }

                // Concatenate Ct to the right end of Nt
                Nt.conservativeResize(Nt.rows(), Nt.cols() + Ct.cols());
                Nt.rightCols(Ct.cols()) = Ct;
                Nt.makeCompressed();
	      }
	  }



	// Set up linear system
	SparseMatrixX<T> Mat(Nt.rows(), Nt.rows()); // Mat will be the matrix on the LHS

	if (!initializer_checkpoint_plan.bypass_data_solve)
	  {
	#ifdef MFA_TBB  // TBB version
	    Eigen::SparseMatrix<T, Eigen::ColMajor> NCol = Nt.transpose();

	    int ntn_sparsity = (2*mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
	    MatProdThreaded(Nt, NCol, Mat, ntn_sparsity);
	#else
	    Mat = Nt * Nt.transpose();
	#endif
	  }
	ApplyNSTraceAwareRidge(Mat, trace_aware_ls_config);

	BaselineCheckpointMetadata baseline_metadata;
	if (baseline_config.baseline_requested)
	  {
	    std::ostringstream lattice;
	    std::ostringstream degree;
	    for (int k = 0; k < dom_dim; ++k)
	      {
		if (k != 0)
		  {
		    lattice << ",";
		    degree << ",";
		  }
		lattice << t.nctrl_pts(k);
		degree << mfa_data.p(k);
	      }

	    baseline_metadata.case_name = "ns_lid_cavity";
	    baseline_metadata.kind = baseline_config.kind;
	    baseline_metadata.alpha1 = baseline_config.alpha1;
	    baseline_metadata.alpha2 = baseline_config.alpha2;
	    if (baseline_config.kind == BaselineKind::data_only ||
	        (baseline_config.alpha1 == 0.0 &&
	         baseline_config.alpha2 == 0.0))
	      baseline_metadata.derivative_orders = "none";
	    else if (baseline_config.alpha1 > 0.0 &&
	             baseline_config.alpha2 > 0.0)
	      baseline_metadata.derivative_orders = "1,2";
	    else if (baseline_config.alpha1 > 0.0)
	      baseline_metadata.derivative_orders = "1";
	    else
	      baseline_metadata.derivative_orders = "2";
	    baseline_metadata.derivative_dims = "x,y,t";
	    baseline_metadata.lattice = lattice.str();
	    baseline_metadata.degree = degree.str();
	    baseline_metadata.initializer_role_sha256 =
	      baseline_config.role_sha256;
	    baseline_metadata.pressure_regime = "zero";
	    baseline_metadata.field_order = "u,v,p";
	    baseline_metadata.coefficient_count_per_field =
	      static_cast<size_t>(num_reg_conds);
	    baseline_metadata.coefficient_count_total =
	      static_cast<size_t>(3) *
	      static_cast<size_t>(num_reg_conds);
	    baseline_metadata.code_sha256 =
	      baseline_config.code_sha256;
	  }

	// Solve the normal equations only when no validated checkpoint supplies the
	// Stage-B start.  Corrected baseline checkpoints use their separate V2
	// contract; the historical Stage-A solve remains unchanged otherwise.
	Eigen::ConjugateGradient<
	  SparseMatrixX<T>,
	  Eigen::Lower|Eigen::Upper> solverr;
	solverr.setTolerance(1e-12);
	if (baseline_config.baseline_requested)
	  solverr.setMaxIterations(
	    std::max(1000, 20 * num_reg_conds));

	int solver_iterations_u = -1;
	int solver_iterations_v = -1;
	double solver_residual_u =
	  std::numeric_limits<double>::quiet_NaN();
	double solver_residual_v =
	  std::numeric_limits<double>::quiet_NaN();
	std::string solver_status = "not_run";

	if (initializer_checkpoint_plan.bypass_data_solve)
	  {
	    if (t.ctrl_pts.rows() != num_reg_conds ||
	        t.ctrl_pts.cols() != 1 ||
	        t2.ctrl_pts.rows() != num_reg_conds ||
	        t2.ctrl_pts.cols() != 1 ||
	        t3.ctrl_pts.rows() != num_reg_conds ||
	        t3.ctrl_pts.cols() != 1)
	      throw std::runtime_error(
	        "NS initializer checkpoint requires scalar u, v, and p fields");
	    for (int i = 0; i < num_reg_conds; ++i)
	      {
	        t.ctrl_pts(i, 0) = static_cast<T>(
	          initializer_checkpoint_payload[static_cast<size_t>(i)]);
	        t2.ctrl_pts(i, 0) = static_cast<T>(
	          initializer_checkpoint_payload[
	            static_cast<size_t>(num_reg_conds + i)]);
	        t3.ctrl_pts(i, 0) = static_cast<T>(
	          initializer_checkpoint_payload[
	            static_cast<size_t>(2 * num_reg_conds + i)]);
	      }
	    solver_status = "checkpoint_loaded_data_solve_skipped";
	    fmt::print(
	      "PIMFA_NS_INITIALIZER source=checkpoint action=data_solve_skipped "
	      "path={} coefficient_count={} pressure_regime={}\n",
	      initializer_checkpoint_plan.path,
	      initializer_checkpoint_payload.size(),
	      pressure_checkpoint_regime);
	  }
	else if (baseline_config.baseline_requested &&
	    !baseline_config.checkpoint_in.empty())
	  {
	    const std::vector<double> payload =
	      ReadBaselineCheckpointV2(
	        baseline_config.checkpoint_in,
	        baseline_metadata);
	    if (t.ctrl_pts.rows() != num_reg_conds ||
	        t.ctrl_pts.cols() != 1 ||
	        t2.ctrl_pts.rows() != num_reg_conds ||
	        t2.ctrl_pts.cols() != 1 ||
	        t3.ctrl_pts.rows() != num_reg_conds ||
	        t3.ctrl_pts.cols() != 1)
	      throw std::runtime_error(
	        "NS baseline checkpoint requires scalar u, v, and p fields");
	    for (int i = 0; i < num_reg_conds; ++i)
	      {
		t.ctrl_pts(i, 0) = static_cast<T>(
		  payload[static_cast<size_t>(i)]);
		t2.ctrl_pts(i, 0) = static_cast<T>(
		  payload[
		    static_cast<size_t>(num_reg_conds + i)]);
		const double pressure =
		  payload[
		    static_cast<size_t>(2 * num_reg_conds + i)];
		if (pressure != 0.0 || std::signbit(pressure))
		  throw std::runtime_error(
		    "NS baseline checkpoint pressure payload must be exact +0");
		t3.ctrl_pts(i, 0) = T(0);
	      }
	    solver_status = "checkpoint_loaded";
	    fmt::print(
	      "PIMFA_BASELINE_CHECKPOINT action=loaded path={} "
	      "coefficient_count={}\n",
	      baseline_config.checkpoint_in,
	      payload.size());
	  }
	else if (baseline_config.baseline_requested)
	  {
	    solverr.compute(Mat);
	    if (solverr.info() != Eigen::Success)
	      throw std::runtime_error(
	        "Corrected NS baseline matrix factorization failed");

	    t.ctrl_pts = solverr.solve(R);
	    solver_iterations_u = solverr.iterations();
	    solver_residual_u = solverr.error();
	    if (solverr.info() != Eigen::Success)
	      throw std::runtime_error(
	        "Corrected NS baseline u solve did not converge");

	    t2.ctrl_pts = solverr.solve(R2);
	    solver_iterations_v = solverr.iterations();
	    solver_residual_v = solverr.error();
	    if (solverr.info() != Eigen::Success)
	      throw std::runtime_error(
	        "Corrected NS baseline v solve did not converge");

	    t3.ctrl_pts =
	      MatrixX<T>::Zero(num_reg_conds, pt_dim);
	    solver_status = "success";
	    std::cout
	      << "NS pressure initializer: exact zero "
	      << "(volumetric pressure withheld)\n";
	  }
	else if (trace_aware_ls_config.enabled)
	  {
	    if (t.ctrl_pts.rows() != num_reg_conds ||
	        t.ctrl_pts.cols() != 1 ||
	        t2.ctrl_pts.rows() != num_reg_conds ||
	        t2.ctrl_pts.cols() != 1 ||
	        t3.ctrl_pts.rows() != num_reg_conds ||
	        t3.ctrl_pts.cols() != 1)
	      throw std::runtime_error(
	        "Trace-aware NS initializer requires scalar u, v, and p fields");

	    solverr.compute(Mat);
	    if (solverr.info() != Eigen::Success)
	      throw std::runtime_error(
	        "Trace-aware NS initializer matrix factorization failed");

	    t.ctrl_pts = solverr.solve(R);
	    const Eigen::ComputationInfo status_u = solverr.info();
	    solver_iterations_u = solverr.iterations();
	    solver_residual_u = solverr.error();
	    const double coefficient_max_u =
	      ValidateNSTraceAwareFieldSolve(
	        "u",
	        status_u,
	        solver_residual_u,
	        solver_iterations_u,
	        t.ctrl_pts);

	    t2.ctrl_pts = solverr.solve(R2);
	    const Eigen::ComputationInfo status_v = solverr.info();
	    solver_iterations_v = solverr.iterations();
	    solver_residual_v = solverr.error();
	    const double coefficient_max_v =
	      ValidateNSTraceAwareFieldSolve(
	        "v",
	        status_v,
	        solver_residual_v,
	        solver_iterations_v,
	        t2.ctrl_pts);

	    t3.ctrl_pts = MatrixX<T>::Zero(num_reg_conds, pt_dim);
	    RequireNSTraceAwarePositiveZeroPressure(t3.ctrl_pts);
	    solver_status = "success";
	    fmt::print(
	      "PIMFA_NS_TRACE_AWARE_LS status=success alpha={:.17g} "
	      "active_rows={} coefficient_count_per_field={} "
	      "status_u=success residual_u={:.17g} iterations_u={} "
	      "coefficient_max_u={:.17g} "
	      "status_v=success residual_v={:.17g} iterations_v={} "
	      "coefficient_max_v={:.17g} pressure=positive_zero\n",
	      trace_aware_ls_config.alpha,
	      n_train,
	      num_reg_conds,
	      solver_residual_u,
	      solver_iterations_u,
	      coefficient_max_u,
	      solver_residual_v,
	      solver_iterations_v,
	      coefficient_max_v);
	  }
	else
	  {
	    solverr.compute(Mat);
	    if (solverr.info() != Eigen::Success)
	      fmt::print("WARNING: Matrix decomposition failed in EncodeTensor\n");
	    else if (verbose >= 2)
	      fmt::print("Sparse matrix factorization successful\n");

	    t.ctrl_pts = solverr.solve(R);
	    t2.ctrl_pts = solverr.solve(R2);
	    if (pressure_init_uses_data)
	      {
		t3.ctrl_pts = solverr.solve(R3);
		std::cout
		  << "NS pressure initializer: full pressure data are enabled\n";
	      }
	    else
	      {
		// Do not leak the withheld volumetric pressure field through Stage A.
		// Permitted pressure IC/BC information is introduced only by the
		// explicitly weighted Stage-B terms.
		t3.ctrl_pts =
		  MatrixX<T>::Zero(t3.nctrl_pts.prod(), pt_dim);
		std::cout
		  << "NS pressure initializer: zero "
		  << "(volumetric pressure withheld)\n";
	      }
	    if (solverr.info() != Eigen::Success)
	      {
		fmt::print("WARNING: Least-squares solve failed in EncodeTensor\n");
		fmt::print(
		  "  error: {} (tolerance = {})\n",
		  solverr.error(),
		  solverr.tolerance());
		fmt::print(
		  "  # iterations: {} (max iterations = {})\n",
		  solverr.iterations(),
		  solverr.maxIterations());
	      }
	    else if (verbose >= 2)
	      {
		fmt::print("Sparse matrix solve successful\n");
		fmt::print(
		  "  # iterations: {}\n",
		  solverr.iterations());
	      }
	  }

	if (baseline_config.baseline_requested)
	  {
	    std::vector<double> payload(
	      baseline_metadata.coefficient_count_total);
	    size_t pressure_nonzero_count = 0;
	    size_t pressure_negative_zero_count = 0;
	    double pressure_l2_squared = 0.0;
	    double pressure_max_abs = 0.0;
	    for (int i = 0; i < num_reg_conds; ++i)
	      {
		payload[static_cast<size_t>(i)] =
		  static_cast<double>(t.ctrl_pts(i, 0));
		payload[
		  static_cast<size_t>(num_reg_conds + i)] =
		  static_cast<double>(t2.ctrl_pts(i, 0));
		const double pressure =
		  static_cast<double>(t3.ctrl_pts(i, 0));
		payload[
		  static_cast<size_t>(2 * num_reg_conds + i)] =
		  pressure;
		if (pressure != 0.0)
		  ++pressure_nonzero_count;
		if (pressure == 0.0 && std::signbit(pressure))
		  ++pressure_negative_zero_count;
		pressure_l2_squared += pressure * pressure;
		pressure_max_abs =
		  std::max(pressure_max_abs, std::abs(pressure));
	      }
	    if (pressure_nonzero_count != 0 ||
	        pressure_negative_zero_count != 0)
	      throw std::runtime_error(
	        "Corrected NS baseline failed the exact +0 pressure audit");

	    if (!baseline_config.checkpoint_out.empty())
	      {
		WriteBaselineCheckpointV2(
		  baseline_config.checkpoint_out,
		  baseline_metadata,
		  payload);
		fmt::print(
		  "PIMFA_BASELINE_CHECKPOINT action=written path={} "
		  "coefficient_count={}\n",
		  baseline_config.checkpoint_out,
		  payload.size());
	      }

	    const MatrixX<T> prediction_u =
	      Nt_data.transpose() * t.ctrl_pts;
	    const MatrixX<T> prediction_v =
	      Nt_data.transpose() * t2.ctrl_pts;
	    T data_sse_u = T(0);
	    T data_sse_v = T(0);
	    VectorX<T> target_u(mfa_data.dim());
	    VectorX<T> target_v(mfa_data2.dim());
	    size_t audited_rows = 0;
	    for (auto input_it = input.begin();
	         input_it != input.end();
	         ++input_it)
	      {
		bool active = false;
		for (typename SparseMatrixX<T>::InnerIterator it(
		       Nt_data,
		       input_it.idx());
		     it;
		     ++it)
		  {
		    active = true;
		    break;
		  }
		if (!active)
		  continue;

		input_it.coords(
		  target_u,
		  mfa_data.min_dim,
		  mfa_data.max_dim);
		input_it.coords(
		  target_v,
		  mfa_data2.min_dim,
		  mfa_data2.max_dim);
		for (int component = 0;
		     component < target_u.size();
		     ++component)
		  {
		    const T residual =
		      prediction_u(input_it.idx(), component) -
		      target_u(component);
		    data_sse_u += residual * residual;
		  }
		for (int component = 0;
		     component < target_v.size();
		     ++component)
		  {
		    const T residual =
		      prediction_v(input_it.idx(), component) -
		      target_v(component);
		    data_sse_v += residual * residual;
		  }
		++audited_rows;
	      }
	    if (audited_rows != n_train)
	      throw std::runtime_error(
	        "NS Reg-MFA objective audit active-row count mismatch");

	    const double unavailable =
	      std::numeric_limits<double>::quiet_NaN();
	    double roughness1_u = unavailable;
	    double roughness1_v = unavailable;
	    double roughness2_u = unavailable;
	    double roughness2_v = unavailable;
	    if (baseline_config.alpha1 > 0.0)
	      {
		roughness1_u = static_cast<double>(
		  (derivative1_transpose.transpose() *
		   t.ctrl_pts).squaredNorm());
		roughness1_v = static_cast<double>(
		  (derivative1_transpose.transpose() *
		   t2.ctrl_pts).squaredNorm());
	      }
	    if (baseline_config.alpha2 > 0.0)
	      {
		roughness2_u = static_cast<double>(
		  (derivative2_transpose.transpose() *
		   t.ctrl_pts).squaredNorm());
		roughness2_v = static_cast<double>(
		  (derivative2_transpose.transpose() *
		   t2.ctrl_pts).squaredNorm());
	      }

	    const double data_sse =
	      static_cast<double>(data_sse_u + data_sse_v);
	    const double data_mean =
	      data_sse / static_cast<double>(n_train);
	    const double derivative_rows =
	      static_cast<double>(dom_dim) *
	      static_cast<double>(num_reg_conds);
	    const double roughness1_mean =
	      baseline_config.alpha1 > 0.0
	        ? (roughness1_u + roughness1_v) /
	          derivative_rows
	        : unavailable;
	    const double roughness2_mean =
	      baseline_config.alpha2 > 0.0
	        ? (roughness2_u + roughness2_v) /
	          derivative_rows
	        : unavailable;
	    double objective = data_mean;
	    if (baseline_config.alpha1 > 0.0)
	      objective +=
	        baseline_config.alpha1 * roughness1_mean;
	    if (baseline_config.alpha2 > 0.0)
	      objective +=
	        baseline_config.alpha2 * roughness2_mean;

	    fmt::print(
	      "PIMFA_BASELINE_MATRIX n_train={} nctrl={} "
	      "data_transpose_rows={} data_transpose_cols={} "
	      "data_nnz={} augmented_cols={} augmented_nnz={} "
	      "derivative1_rows={} derivative1_nnz={} "
	      "derivative2_rows={} derivative2_nnz={} "
	      "derivative1_scale={:.17g} derivative2_scale={:.17g} "
	      "operator_application=u,v pressure_application=none\n",
	      n_train,
	      num_reg_conds,
	      Nt_data.rows(),
	      Nt_data.cols(),
	      Nt_data.nonZeros(),
	      Nt.cols(),
	      Nt.nonZeros(),
	      derivative1_transpose.cols(),
	      derivative1_transpose.nonZeros(),
	      derivative2_transpose.cols(),
	      derivative2_transpose.nonZeros(),
	      static_cast<double>(derivative1_scale),
	      static_cast<double>(derivative2_scale));
	    fmt::print(
	      "PIMFA_BASELINE_SOLVER status={} tolerance={:.17g} "
	      "max_iterations={} "
	      "iterations_u={} residual_u={:.17g} "
	      "iterations_v={} residual_v={:.17g}\n",
	      solver_status,
	      solverr.tolerance(),
	      solverr.maxIterations(),
	      solver_iterations_u,
	      solver_residual_u,
	      solver_iterations_v,
	      solver_residual_v);
	    fmt::print(
	      "PIMFA_BASELINE_OBJECTIVE data_sse_u={:.17g} "
	      "data_sse_v={:.17g} data_sse={:.17g} "
	      "data_mean={:.17g} roughness1_u={:.17g} "
	      "roughness1_v={:.17g} roughness1_mean={:.17g} "
	      "roughness2_u={:.17g} roughness2_v={:.17g} "
	      "roughness2_mean={:.17g} alpha1={:.17g} "
	      "alpha2={:.17g} objective={:.17g}\n",
	      static_cast<double>(data_sse_u),
	      static_cast<double>(data_sse_v),
	      data_sse,
	      data_mean,
	      roughness1_u,
	      roughness1_v,
	      roughness1_mean,
	      roughness2_u,
	      roughness2_v,
	      roughness2_mean,
	      baseline_config.alpha1,
	      baseline_config.alpha2,
	      objective);
	    fmt::print(
	      "PIMFA_BASELINE_PRESSURE regime=zero "
	      "coefficient_count={} nonzero_count={} "
	      "negative_zero_count={} l2={:.17g} max_abs={:.17g}\n",
	      num_reg_conds,
	      pressure_nonzero_count,
	      pressure_negative_zero_count,
	      std::sqrt(pressure_l2_squared),
	      pressure_max_abs);
	    fmt::print(
	      "NS physics-informed optimization skipped "
	      "(corrected baseline initializer-only mode)\n");
	    return;
	  }

	double nu = 0.001;
	double lambda =0.1;
	using FunctorType = mfa::ObjectiveFunction_LBFGS<T>;
	using VectorType  = typename FunctorType::VectorType;
	const int n = t.nctrl_pts.prod()*3;
	Eigen::VectorXd P    = VectorType::Ones(n);
	double          fx   = 0.0f;
	int halfSize = P.size()/3;
	LBFGSpp::LBFGSBParam<double> param;
	param.epsilon = readEnv<double>("MFA_OPTIMIZER_EPSILON", 1e-5);
		param.epsilon_rel =
		  readEnv<double>("MFA_OPTIMIZER_EPSILON_REL", 1e-5);
		param.past =
		  readEnv<int>("MFA_OPTIMIZER_PAST", param.past);
		param.delta =
		  readEnv<double>("MFA_OPTIMIZER_DELTA", param.delta);
		param.max_iterations =
		  readEnv<int>("MFA_MAX_ITERATIONS", param.max_iterations);
	LBFGSpp::LBFGSBSolver<double> solver(param);
        FunctorType fun(n,lambda,nu,mfa_data,mfa_data2,mfa_data3, input,input_IC,input_BC,input_BCx,input_BCy,verbose);

	// Bounds
	VectorXd lb = VectorXd::Constant(n, -initializer_coefficient_bound);
	VectorXd ub = VectorXd::Constant(n, initializer_coefficient_bound);

	// Initial guess

	P.segment(0,halfSize)=t.ctrl_pts;
	P.segment(halfSize,halfSize)=t2.ctrl_pts;
	P.segment(2*halfSize,halfSize)=t3.ctrl_pts;

	const bool hard_trace_enabled =
	  nsIntegralStrictBoolean("MFA_NS_HARD_TRACE_LOCK", false);
	NSHardTraceBoundsAttestation hard_trace_attestation;
	VectorXd hard_trace_reference;
	if (hard_trace_enabled)
	  {
	    if (!initializer_checkpoint_plan.bypass_data_solve)
	      throw std::invalid_argument(
	        "MFA_NS_HARD_TRACE_LOCK=1 requires MFA_INIT_CHECKPOINT_IN");
	    if (pressure_checkpoint_regime != "zero")
	      throw std::invalid_argument(
	        "MFA_NS_HARD_TRACE_LOCK=1 requires the zero-pressure "
	        "initializer regime");
	    if (dom_dim != 3 || pt_dim != 1 ||
	        mfa_data.tmesh.tensor_prods.size() != 1 ||
	        mfa_data2.tmesh.tensor_prods.size() != 1 ||
	        mfa_data3.tmesh.tensor_prods.size() != 1 ||
	        t.ctrl_pts.cols() != 1 || t2.ctrl_pts.cols() != 1 ||
	        t3.ctrl_pts.cols() != 1)
	      throw std::invalid_argument(
	        "Hard-trace TGV follow-up requires one scalar x-y-t tensor "
	        "for each of u, v, and p");
	    if (t.nctrl_pts.size() != 3 ||
	        t.nctrl_pts(0) != 20 || t.nctrl_pts(1) != 20 ||
	        t.nctrl_pts(2) != 50 ||
	        (t2.nctrl_pts.array() != t.nctrl_pts.array()).any() ||
	        (t3.nctrl_pts.array() != t.nctrl_pts.array()).any())
	      throw std::invalid_argument(
	        "Hard-trace TGV follow-up requires the identical 20x20x50 "
	        "u/v/p control lattice");
	    for (int axis = 0; axis < 3; ++axis)
	      {
	        if (mfa_data.p(axis) != 3 || mfa_data2.p(axis) != 3 ||
	            mfa_data3.p(axis) != 3 ||
	            mfa_data2.tmesh.all_knots[axis] !=
	              mfa_data.tmesh.all_knots[axis] ||
	            mfa_data3.tmesh.all_knots[axis] !=
	              mfa_data.tmesh.all_knots[axis])
	          throw std::invalid_argument(
	            "Hard-trace TGV follow-up requires identical cubic u/v/p "
	            "spline spaces");
	        const std::vector<T>& knots =
	          mfa_data.tmesh.all_knots[axis];
	        const int controls = t.nctrl_pts(axis);
	        if (knots.size() != static_cast<size_t>(controls + 4))
	          throw std::invalid_argument(
	            "Hard-trace TGV follow-up knot-vector size mismatch");
	        const T span = T(1) / static_cast<T>(controls - 3);
	        const T tolerance = T(512) *
	          std::numeric_limits<T>::epsilon();
	        for (int knot = 0; knot < controls + 4; ++knot)
	          {
	            const T expected = knot <= 3 ? T(0) :
	              (knot >= controls ? T(1) :
	               static_cast<T>(knot - 3) * span);
	            if (!std::isfinite(knots[static_cast<size_t>(knot)]) ||
	                std::abs(knots[static_cast<size_t>(knot)] - expected) >
	                  tolerance)
	              throw std::invalid_argument(
	                "Hard-trace TGV follow-up requires open-uniform knots");
	          }
	      }
	    for (int i = 0; i < halfSize; ++i)
	      {
	        const double pressure = P(2 * halfSize + i);
	        if (pressure != 0.0 || std::signbit(pressure))
	          throw std::invalid_argument(
	            "Hard-trace TGV follow-up requires exact +0 Stage-A "
	            "pressure coefficients");
	      }
	    hard_trace_reference = P;
	    hard_trace_attestation =
	      ApplyNSHardTraceBounds(P, t.nctrl_pts, lb, ub);
	    if (hard_trace_attestation.locked_u != 4124 ||
	        hard_trace_attestation.locked_v != 4124 ||
	        hard_trace_attestation.locked_p != 0 ||
	        hard_trace_attestation.locked_indices.size() != 8248 ||
	        hard_trace_attestation.free_total != 51752)
	      throw std::runtime_error(
	        "Hard-trace TGV follow-up lock-count attestation failed");
	    fmt::print(
	      "PIMFA_NS_HARD_TRACE_BOUNDS status=applied "
	      "locked_u={} locked_v={} locked_p={} locked_total={} "
	      "free_total={} lattice=20x20x50 degree=3 pressure=positive_zero\n",
	      hard_trace_attestation.locked_u,
	      hard_trace_attestation.locked_v,
	      hard_trace_attestation.locked_p,
	      hard_trace_attestation.locked_indices.size(),
	      hard_trace_attestation.free_total);
	  }

        const char* checkpoint_out_env = std::getenv("MFA_INIT_CHECKPOINT_OUT");
        const std::string checkpoint_out =
          checkpoint_out_env ? checkpoint_out_env : "mfa_data_only_init.txt";
        if (!checkpoint_out.empty())
        {
          std::ofstream stream(checkpoint_out.c_str());
          if (!stream)
            throw std::runtime_error(
                std::string("Cannot write MFA_INIT_CHECKPOINT_OUT: ") + checkpoint_out);
          stream << "# PIMFA_NS_INIT_V1 pressure=" << pressure_checkpoint_regime
                 << " n=" << P.size() << "\n";
          stream << std::setprecision(std::numeric_limits<T>::max_digits10);
          for (int i = 0; i < P.size(); ++i)
            stream << P(i) << "\n";
          std::cout << "Saved reusable NS initializer to " << checkpoint_out << "\n";
        }

        if (readEnv<int>("MFA_PREFILTERED_PDE_BENCHMARK", 0) != 0)
        {
          const int evaluations =
            readEnv<int>("MFA_PREFILTERED_PDE_EVALUATIONS", 5);
          if (evaluations < 1)
            throw std::invalid_argument(
              "MFA_PREFILTERED_PDE_EVALUATIONS must be positive");
          VectorType benchmark_gradient(P.size());
          const std::chrono::steady_clock::time_point started =
            std::chrono::steady_clock::now();
          for (int evaluation = 0; evaluation < evaluations; ++evaluation)
            fx = fun.evaluatePrefilteredPde(P, benchmark_gradient);
          const double evaluation_seconds =
            std::chrono::duration<double>(
              std::chrono::steady_clock::now() - started).count();
          std::cout
            << "PIMFA_PREFILTERED_PDE_RESULT case=ns evaluations="
            << evaluations
            << " active_rows=" << fun.prefilteredPdeRows()
            << " construction_seconds="
            << std::setprecision(17)
            << fun.prefilteredConstructionSeconds()
            << " evaluation_seconds=" << evaluation_seconds
            << " loss=" << fx
            << " gradient_norm=" << benchmark_gradient.norm()
            << "\n";
          return;
        }

        const bool gradient_decomposition_only =
          nsIntegralStrictBoolean(
            "MFA_NS_GRADIENT_DECOMPOSITION_ONLY", false);
        const char* gradient_vector_out =
          std::getenv("MFA_NS_GRADIENT_DECOMPOSITION_VECTOR_OUT");
        const char* gradient_term_out =
          std::getenv("MFA_NS_GRADIENT_DECOMPOSITION_TERM_OUT");
        if (gradient_decomposition_only)
        {
          if (!initializer_checkpoint_plan.bypass_data_solve)
            throw std::invalid_argument(
              "MFA_NS_GRADIENT_DECOMPOSITION_ONLY=1 requires "
              "MFA_INIT_CHECKPOINT_IN");
          if (!gradient_vector_out || !*gradient_vector_out ||
              !gradient_term_out || !*gradient_term_out)
            throw std::invalid_argument(
              "Gradient decomposition requires both "
              "MFA_NS_GRADIENT_DECOMPOSITION_VECTOR_OUT and "
              "MFA_NS_GRADIENT_DECOMPOSITION_TERM_OUT");
          fun.writeObjectiveGradientAudit(
            P, gradient_vector_out, gradient_term_out);
          t.ctrl_pts = P.segment(0, halfSize);
          t2.ctrl_pts = P.segment(halfSize, halfSize);
          t3.ctrl_pts = P.segment(2 * halfSize, halfSize);
          std::cout
            << "MFA_NS_GRADIENT_DECOMPOSITION_ONLY=1: "
            << "completed no-optimization production-objective audit\n";
          return;
        }
        if ((gradient_vector_out && *gradient_vector_out) ||
            (gradient_term_out && *gradient_term_out))
          throw std::invalid_argument(
            "Gradient-decomposition output paths require "
            "MFA_NS_GRADIENT_DECOMPOSITION_ONLY=1");
        const int gradient_checks = readEnv<int>("MFA_CHECK_GRADIENT", 0);
        if (gradient_checks > 0)
        {
          const T gradient_step =
            readEnv<T>("MFA_GRADIENT_CHECK_STEP", T(1.0e-6));
          const T gradient_tolerance =
            readEnv<T>("MFA_GRADIENT_CHECK_TOL", T(1.0e-5));
          if (!fun.checkDirectionalGradients(
                  P, gradient_checks, gradient_step, gradient_tolerance))
            throw std::runtime_error(
                "NS directional gradient check failed; refusing to optimize");
          if (nsIntegralStrictBoolean(
                "MFA_GRADIENT_CHECK_PERTURBED", false))
          {
            std::mt19937_64 perturb_generator(20260729u);
            std::normal_distribution<T> perturb_normal(T(0), T(1));
            VectorType perturbed = P;
            VectorType perturbation(P.size());
            for (int i = 0; i < perturbation.size(); ++i)
              perturbation(i) = perturb_normal(perturb_generator);
            perturbation.normalize();
            const T perturbation_scale =
              nsIntegralStrictScalar<T>(
                "MFA_GRADIENT_PERTURBATION", T(1.0e-3)) *
              std::max<T>(T(1), P.norm());
            perturbed += perturbation_scale * perturbation;
            if (!fun.checkDirectionalGradients(
                  perturbed, gradient_checks,
                  gradient_step, gradient_tolerance))
              throw std::runtime_error(
                "NS perturbed directional gradient check failed; "
                "refusing to optimize");
          }
          std::cout << "NS directional gradient checks passed\n";
        }

        if (readEnv<int>("MFA_GRADIENT_CHECK_ONLY", 0) != 0)
        {
          t.ctrl_pts = P.segment(0, halfSize);
          t2.ctrl_pts = P.segment(halfSize, halfSize);
          t3.ctrl_pts = P.segment(2 * halfSize, halfSize);
          std::cout << "MFA_GRADIENT_CHECK_ONLY=1: skipping L-BFGS-B optimization\n";
          return;
        }

        if (readEnv<int>("MFA_INITIALIZER_ONLY", 0) != 0)
        {
          t.ctrl_pts = P.segment(0, halfSize);
          t2.ctrl_pts = P.segment(halfSize, halfSize);
          t3.ctrl_pts = P.segment(2 * halfSize, halfSize);
          std::cout
            << "MFA_INITIALIZER_ONLY=1: saved/loaded Stage-A coefficients "
            << "and skipped physics-informed optimization\n";
          return;
        }

	int niter = solver.minimize(fun, P, fx, lb, ub);
	if (hard_trace_enabled)
	  {
	    const double locked_max_abs_drift =
	      NSHardTraceMaximumDrift(
	        P, hard_trace_reference, hard_trace_attestation.locked_indices);
	    if (locked_max_abs_drift != 0.0)
	      throw std::runtime_error(
	        "Hard-trace equality-bound coefficient drift is nonzero");
	    fmt::print(
	      "PIMFA_NS_HARD_TRACE_FINAL status=pass locked_total={} "
	      "locked_max_abs_drift={:.17g}\n",
	      hard_trace_attestation.locked_indices.size(),
	      locked_max_abs_drift);
	  }

	const double initial_proj_grad = solver.initial_grad_norm();
	const double final_proj_grad = solver.final_grad_norm();
	const int accepted_updates = solver.update_count();
	const double relative_threshold = param.epsilon_rel * P.norm();
	const char* termination =
	  (param.max_iterations != 0 && niter >= param.max_iterations)
	  ? "max_iterations"
	  : (final_proj_grad <= param.epsilon)
	    ? "projected_gradient_absolute"
	    : (final_proj_grad <= relative_threshold)
	      ? "projected_gradient_relative"
	      : "other";
	std::cout << niter << " iterations" << std::endl;
	std::cout << "LBFGS initial projected gradient norm: "
	          << std::setprecision(17) << initial_proj_grad << std::endl;
	std::cout << "LBFGS accepted update count: "
	          << accepted_updates << std::endl;
	std::cout << "LBFGS final projected gradient norm: "
	          << std::setprecision(17) << final_proj_grad << std::endl;
	std::cout << "LBFGS epsilon absolute: "
	          << std::setprecision(17) << param.epsilon << std::endl;
	std::cout << "LBFGS epsilon relative: "
	          << std::setprecision(17) << param.epsilon_rel << std::endl;
	std::cout << "LBFGS relative threshold: "
	          << std::setprecision(17) << relative_threshold << std::endl;
	std::cout << "LBFGS past objective values: "
	          << param.past << std::endl;
	std::cout << "LBFGS objective delta tolerance: "
	          << std::setprecision(17) << param.delta << std::endl;
	std::cout << "LBFGS max iterations: "
	          << param.max_iterations << std::endl;
	std::cout << "LBFGS termination: " << termination << std::endl;

        if (readEnv<int>("MFA_FINAL_STATE_HANDSHAKE", 0) != 0)
        {
          VectorType final_state_gradient(P.size());
          const double final_state_objective =
            fun(P, final_state_gradient);
          const char* final_checkpoint_path =
            std::getenv("MFA_FINAL_COEFFICIENT_CHECKPOINT_OUT");
          if (!final_checkpoint_path || !*final_checkpoint_path)
            throw std::invalid_argument(
              "MFA_FINAL_STATE_HANDSHAKE=1 requires "
              "MFA_FINAL_COEFFICIENT_CHECKPOINT_OUT");
          std::ofstream final_checkpoint(final_checkpoint_path);
          if (!final_checkpoint)
            throw std::runtime_error(
              std::string("Cannot write final coefficient checkpoint: ") +
              final_checkpoint_path);
          final_checkpoint
            << "# PIMFA_NS_FINAL_COEFFICIENTS_V1 n=" << P.size() << "\n"
            << std::setprecision(std::numeric_limits<double>::max_digits10);
          for (int i = 0; i < P.size(); ++i)
            final_checkpoint << P(i) << "\n";
          final_checkpoint.close();
          if (!final_checkpoint)
            throw std::runtime_error(
              std::string("Failed while writing final coefficient checkpoint: ") +
              final_checkpoint_path);
          std::cout
            << "PIMFA_NS_FINAL_STATE"
            << " handshake=1"
            << " objective=" << std::setprecision(17)
            << final_state_objective
            << " gradient_l2=" << final_state_gradient.norm()
            << " coefficient_count=" << P.size()
            << " coefficient_l2=" << P.norm()
            << " checkpoint=" << final_checkpoint_path
            << std::endl;
        }
	int nP = P.size()/3;

	t.ctrl_pts = P.segment(0,nP);
	t2.ctrl_pts = P.segment(nP,nP);
	t3.ctrl_pts = P.segment(2*nP,nP);

      }

#ifdef MFA_TMESH

        // whether a curve of input points intersects a tensor product
        // only checks dimensions above the current dim
        // higher dims are for input point space, lower dims are in control point space
        // lower dims are guranteed to intersect already
        // NB: not general purpose because does not check all dims
        bool CurveIntersectsTensor(
                TensorIdx               t_idx,                          // index of current tensor
                int                     dim,                            // current dimension of curve
                const VectorXi&         nin_pts,                        // number of input points
                const VectorXi&         start_ijk)                      // i,j,k of start of input points
        {
            auto& t = mfa_data.tmesh.tensor_prods[t_idx];

            // ijk and param of point in the middle of the curve
            VectorXi mid_ijk = start_ijk;
            mid_ijk(dim) += nin_pts(dim) / 2;
            VectorX<T> mid_param(dom_dim);

            for (auto i = dim + 1; i < dom_dim; i++)           // only checks dimensions after current dim; earlier dims guranteed to intersect
            {
                mid_param(i) = input.params->param_grid[i][mid_ijk(i)];
                if (mid_param(i) < mfa_data.tmesh.all_knots[i][t.knot_mins[i]] || mid_param(i) > mfa_data.tmesh.all_knots[i][t.knot_maxs[i]])
                    return false;
            }

            return true;
        }

        // computes curve of free control points in one dimension
        // helper function for EncodeTensorLocalSeparable
        void ComputeCtrlPtCurve(
                CurveIterator&          in_curve_iter,                  // current curve
                TensorIdx               t_idx,                          // index of current tensor
                int                     dim,                            // current curve dimension
                MatrixX<T>&             R,                              // right hand side, allocated by caller
                MatrixX<T>&             Q,                              // first matrix of input points, allocated by caller
                MatrixX<T>&             Q1,                             // second matrix of input points, allocated by caller
                MatrixX<T>&             Nfree,                          // free basis functions, allocated by caller
                MatrixX<T>&             Ncons,                          // constraint basis functions, allocated by caller
                MatrixX<T>&             Pcons,                          // constraint control points, allocated by caller
                MatrixX<T>&             P,                              // (output) solution control points, allocated by caller
                ConsType                cons_type,                      // constraint type
                const VectorXi&         nin_pts,                        // number of input points
                const VectorXi&         start_ijk,                      // i,j,k of start of input points
                int                     ncons,                          // number of constraints
                double                  free_time,                      // time to find free basis functions
                double                  cons_time,                      // time to find constraints
                double                  norm_time,                      // time to normalize basis functions
                double                  solve_time)                     // time to solve the matrix
        {
            auto& t = mfa_data.tmesh.tensor_prods[t_idx];

            // copy one curve of input points to right hand side
            while (!in_curve_iter.done())
            {
                if (dim == 0)
                    R.row(in_curve_iter.cur_iter()) =
                        input.domain.block(in_curve_iter.ijk_idx(in_curve_iter.cur_ijk()), mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
                else if (dim % 2 == 0)
                    R.row(in_curve_iter.cur_iter()) = Q.row(in_curve_iter.ijk_idx(in_curve_iter.cur_ijk()));
                else
                    R.row(in_curve_iter.cur_iter()) = Q1.row(in_curve_iter.ijk_idx(in_curve_iter.cur_ijk()));
                in_curve_iter.incr_iter();
            }

            // find matrix of free control point basis functions
            double t0 = MPI_Wtime();
            FreeCtrlPtCurve(dim, t_idx, start_ijk, nin_pts(dim), Nfree);

            // timing
            free_time   += (MPI_Wtime() - t0);
            t0          = MPI_Wtime();

            // normalize Nfree and Ncons such that the row sum of Nfree + Ncons = 1.0
            t0          = MPI_Wtime();              // timing

#ifdef MFA_TBB

            static affinity_partitioner ap;
            parallel_for (blocked_range<size_t>(0, Nfree.rows()), [&] (blocked_range<size_t>& r)
            {
                for (auto i = r.begin(); i < r.end(); i++)

#else

                for (auto i = 0; i < Nfree.rows(); i++)

#endif

                {
                    bool error = false;
                    T sum = Nfree.row(i).sum();

                    if (cons_type != ConsType::MFA_NO_CONSTRAINT && Ncons.rows() > i)
                        sum += Ncons.row(i).sum();
                    if (sum > 0.0)
                    {
                        Nfree.row(i) /= sum;
                        if (cons_type != ConsType::MFA_NO_CONSTRAINT && Ncons.rows() > i)
                            Ncons.row(i) /= sum;
                    }
                    else
                        fmt::print(stderr, "ComputeCtrlPtCurve(): row {} has 0.0 row sum for free and constraint basis functions in dim {}\n",
                                i, dim);
                }

#ifdef MFA_TBB

            }, ap);

#endif

            norm_time += (MPI_Wtime() - t0);                    // timing

            // R is the right hand side needed for solving N * P = R
            if (Pcons.rows())
                R -= Ncons * Pcons;

            // solve
            t0 = MPI_Wtime();
            P = (Nfree.transpose() * Nfree).ldlt().solve(Nfree.transpose() * R);
            solve_time += (MPI_Wtime() - t0);
        }

        // computes curve of free control points basis functions in one dimension
        // matrix Nfree of basis functions is m rows x n columns, where
        //  m is number of input points covered by constraints and free control points
        //  n is number of free control points
        // the curve of input points may not be in the tensor of control points, but can be in a neighbor if the
        //  curve is in the constraint region in a dim. orthogonal to the current dim.
        // helper function for EncodeTensorLocalSeparable
        void FreeCtrlPtCurve(
                int                 dim,                // current dimension
                TensorIdx           t_idx,              // index of tensor of control points
                const VectorXi&     start_ijk,          // multidim index of first input point for the curve, including constraints
                size_t              npts,               // number of input points in current dim, including constraints
                MatrixX<T>&         Nfree)              // (output) matrix of free control points basis functions
        {
            auto&               t = mfa_data.tmesh.tensor_prods[t_idx];
            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor
            Nfree = MatrixX<T>::Zero(npts, t.nctrl_pts(dim));

            // local knot vector
            vector<KnotIdx> local_knot_idxs(mfa_data.p(dim) + 2);                               // local knot indices for current dim
            vector<T> local_knots(mfa_data.p(dim) + 2);                                         // local knot vector for current dim

            VectorX<T> param(dom_dim);

            // for start of the curve, for dims prior to current dim, find anchor and param
            // those dims are in control point index space for the current tensor
            for (auto i = 0; i < dim; i++)
            {
                mfa_data.tmesh.knot_idx_ofst(t, t.knot_mins[i], start_ijk(i), i, true, anchor[i]);                     // computes anchor as offset from start of tensor
                param(i)    = mfa_data.tmesh.all_knots[i][anchor[i]];
            }

            // for the start of the curve, for current dim. and higher, find param
            // these dims are in the input point index space
            for (auto i = dim; i < dom_dim; i++)
                param(i) = input.params->param_grid[i][start_ijk(i)];

            // for the start of the curve, for higher than the current dim, find anchor
            // these dims are in the input point space
            // in the current dim, the anchor coordinate will be replaced below by the control point anchor
            for (auto i = dim + 1; i < dom_dim; i++)
            {
                // if param == 0, FindSpan finds the last 0-value knot span, but we want the first control point anchor, which is an earlier span
                if (param(i) == 0.0)
                    anchor[i] = (mfa_data.p(i) + 1) / 2;
                else if (param(i) == 1.0)
                    anchor[i] = mfa_data.tmesh.all_knots[i].size() - 2 - (mfa_data.p(i) + 1) / 2;
                else
                    anchor[i] = mfa_data.tmesh.FindSpan(i, param(i), t);
            }

            for (auto i = 0; i < t.nctrl_pts(dim); i++)                                                 // for all control points in current dim
            {
                // anchor of control point in current dim
                anchor[dim] = mfa_data.tmesh.ctrl_pt_anchor_dim(dim, t, i);

                // local knot vector in currrent dimension
                mfa_data.tmesh.knot_intersections_dim(anchor, t_idx, local_knot_idxs, dim);             // local knot indices in current dim
                for (auto n = 0; n < local_knot_idxs.size(); n++)                                       // local knots in only current dim
                    local_knots[n] = mfa_data.tmesh.all_knots[dim][local_knot_idxs[n]];

                for (auto j = 0; j < npts; j++)                                                         // for all input points (for this tensor) in current dim
                {
                    T u = input.params->param_grid[dim][start_ijk(dim) + j];                            // parameter of current input point
                    Nfree(j, i) = mfa_data.OneBasisFun(dim, u, local_knots);                            // basis function
                }
            }       // control points

        }

        // free control points matrix of basis functions
        // matrix Nfree of basis functions is m rows x n columns, where
        // m is number of input points covered by constraints and free control points
        // n is number of free control points
        // helper function for EncodeTensorLocalUnified
        // returns max number of nonzeros in any column
        int  FreeCtrlPtMat(TensorIdx            t_idx,              // index of tensor of control points
                           VectorXi&            ndom_pts,           // number of relevant input points in each dim, covering constraints
                           VectorXi&            dom_starts,         // starting offsets of relevant input points in each dim, covering constraints
                           MatrixX<T>&          Nfree)              // (output) matrix of free control points basis functions
        {
            TensorProduct<T>&   t = mfa_data.tmesh.tensor_prods[t_idx];
            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor
            Nfree = MatrixX<T>::Zero(ndom_pts.prod(), t.ctrl_pts.rows());
            int max_nnz_col = 0;                                                                // max num nonzeros in any column

            fmt::print(stderr, "Nfree has {} rows and {} columns\n", Nfree.rows(), Nfree.cols());

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

#ifdef MFA_TBB  // TBB

            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts()); // iterator over input points
            VolIterator free_iter(t.nctrl_pts);                         // iterator over free control points

            enumerable_thread_specific<vector<vector<KnotIdx>>> thread_local_knot_idxs(dom_dim);   // local knot idices
            enumerable_thread_specific<vector<vector<T>>>       thread_local_knots(dom_dim);       // local knot idices
            enumerable_thread_specific<VectorXi>                thread_dom_ijk(dom_dim);           // multidim index of domain point
            enumerable_thread_specific<VectorXi>                thread_free_ijk(dom_dim);          // multidim index of control point
            enumerable_thread_specific<vector<KnotIdx>>         thread_anchor(dom_dim);            // anchor of control point
            enumerable_thread_specific<vector<int>>             thread_nnz_col(Nfree.cols(), 0);            // number of nonzeros in each column
            static affinity_partitioner                         ap;
            parallel_for (blocked_range2d<size_t>(0, dom_iter.tot_iters(), 0, free_iter.tot_iters()), [&] (blocked_range2d<size_t>& r)
            {
                for (auto i = r.cols().begin(); i < r.cols().end(); i++)                                // for control points
                {
                    if (i == r.cols().begin())
                    {
                        for (auto k = 0; k < dom_dim; k++)
                        {
                            thread_local_knot_idxs.local()[k].resize(mfa_data.p(k) + 2);
                            thread_local_knots.local()[k].resize(mfa_data.p(k) + 2);
                        }
                    }

                    free_iter.idx_ijk(i, thread_free_ijk.local());                                      // ijk of domain point
                    mfa_data.tmesh.ctrl_pt_anchor(t, thread_free_ijk.local(), thread_anchor.local());   // anchor of control point

                    // local knot vector
                    mfa_data.tmesh.knot_intersections(thread_anchor.local(), t_idx, thread_local_knot_idxs.local());
                    for (auto k = 0; k < dom_dim; k++)
                    {
                        for (auto n = 0; n < thread_local_knot_idxs.local()[k].size(); n++)
                            thread_local_knots.local()[k][n] = mfa_data.tmesh.all_knots[k][thread_local_knot_idxs.local()[k][n]];
                    }

                    for (auto j = r.rows().begin(); j < r.rows().end(); j++)                            // for input domain points
                    {
                        dom_iter.idx_ijk(j, thread_dom_ijk.local());                                    // ijk of domain point
                        for (auto k = 0; k < dom_dim; k++)
                        {
                            T u = input.params->param_grid[k][thread_dom_ijk.local()(k)];               // parameter of current input point
                            T B = mfa_data.OneBasisFun(k, u, thread_local_knots.local()[k]);                           // basis function
                            Nfree(j, i) = (k == 0 ? B : Nfree(j, i) * B);
                        }
                        if (Nfree(j, i) != 0.0)
                            thread_nnz_col.local()[i]++;
                    }       // for blocked range rows, ie, input domain points
                }       // for blocked range cols, ie, control points
            }, ap); // parallel for

            // combine thread-safe nnz_col
            vector<int> nnz_col(Nfree.cols(), 0);       // nonzeros in each column of Nfree, summed from threads
            thread_nnz_col.combine_each([&](const vector<int>& nnzs)
            {
                for (auto i = 0; i < Nfree.cols(); i++)
                {
                    nnz_col[i]  += nnzs[i];
                    if (nnz_col[i] > max_nnz_col)
                        max_nnz_col = nnz_col[i];
                }
            });

#else           // serial

            VolIterator free_iter(t.nctrl_pts);                         // iterator over free control points
            while (!free_iter.done())
            {
                VectorXi ijk(dom_dim);                                                 // ijk of current control point
                free_iter.idx_ijk(free_iter.cur_iter(), ijk);

                // anchor of control point
                mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);

                // local knot vector
                mfa_data.tmesh.knot_intersections(anchor, t_idx, local_knot_idxs);
                for (auto k = 0; k < dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                int nnz_col = 0;                                                                // num nonzeros in current column

                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());                     // iterator over input points
                while (!dom_iter.done())
                {
                    for (auto k = 0; k < dom_dim; k++)                                 // for all dims
                    {
                        T u = input.params->param_grid[k][dom_iter.idx_dim(k)];                 // parameter of current input point
                        T B = mfa_data.OneBasisFun(k, u, local_knots[k]);                       // basis function
                        Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) =
                            (k == 0 ? B : Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) * B);
                    }       // for all dims
                    if (Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) != 0.0)
                        nnz_col++;
                    dom_iter.incr_iter();
                }       // domain point iterator

                if (nnz_col > max_nnz_col)
                    max_nnz_col = nnz_col;
                free_iter.incr_iter();
            }       // free control point iterator

#endif          // TBB or serial

            return max_nnz_col;
        }

        // free control points matrix of basis functions
        // helper function for EncodeTensorLocalUnified
        // sparse matrix version; not used for now, but might be in the future
        void FreeCtrlPtMatSparse(TensorIdx          t_idx,              // index of tensor of control points
                                 VectorXi&          ndom_pts,           // number of relevant input points in each dim
                                 VectorXi&          dom_starts,         // starting offsets of relevant input points in each dim
                                 SparseMatrixX<T>&  Nfree_sparse)       // (output) matrix of free control points basis functions
        {
            typedef Eigen::Triplet<T> Triplet;                                                  // (row, column, value)
            vector<Triplet> coeffs;                                                             // nonzero coefficients

            TensorProduct<T>&   t = mfa_data.tmesh.tensor_prods[t_idx];
            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

            // iterator over free control points
            VolIterator free_iter(t.nctrl_pts);
            while (!free_iter.done())
            {
                VectorXi ijk(dom_dim);                                                 // ijk of current control point
                free_iter.idx_ijk(free_iter.cur_iter(), ijk);

                // anchor of control point
                mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);

                // local knot vector
                mfa_data.tmesh.knot_intersections(anchor, t_idx, true, local_knot_idxs);
                for (auto k = 0; k < dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                // iterator over input points
                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());
                while (!dom_iter.done())
                {
                    T v;                                                                        // basis function value
                    for (auto k = 0; k < dom_dim; k++)                                 // for all dims
                    {
                        int p = mfa_data.p(k);                                                  // degree of current dim.
                        T u = input.params->param_grid[k][dom_iter.idx_dim(k)];                             // parameter of current input point
                        T B = mfa_data.OneBasisFun(k, u, local_knots[k]);                       // basis function
                        v = (k == 0 ? B : v * B);
                    }       // for all dims

                    coeffs.push_back(Triplet(dom_iter.cur_iter(), free_iter.cur_iter(), v));
                    dom_iter.incr_iter();
                }       // domain point iterator

                free_iter.incr_iter();
            }       // free control point iterator

            Nfree_sparse.setFromTriplets(coeffs.begin(), coeffs.end());
        }

        // constraint control points matrix of basis functions
        // helper function for EncodeTensorLocalUnified and EncodeTensorLocalSeparable
        // Ncons needs to be sized correctly by caller
        void ConsCtrlPtMat(VectorXi&                ndom_pts,           // number of relevant input points in each dim
                           VectorXi&                dom_starts,         // starting offsets of relevant input points in each dim
                           vector<vector<KnotIdx>>& anchors,            // anchors of constraint control points                                                    // corresponding anchors
                           vector<TensorIdx>&       t_idx_anchors,      // tensors containing corresponding anchors
                           MatrixX<T>&              Ncons)              // (output) matrix of constraint control points basis functions
        {
            Ncons = MatrixX<T>::Constant(ndom_pts.prod(), Ncons.cols(), -1);         // basis functions, -1 means unassigned so far

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

#ifdef MFA_TBB  // TBB

            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());                         // iterator over input points

            enumerable_thread_specific<vector<vector<KnotIdx>>> thread_local_knot_idxs(dom_dim);   // local knot idices
            enumerable_thread_specific<vector<vector<T>>>       thread_local_knots(dom_dim);       // local knot idices
            enumerable_thread_specific<VectorXi>                thread_dom_ijk(dom_dim);   // multidim index of domain point
            static affinity_partitioner                         ap;
            parallel_for (blocked_range2d<size_t>(0, Ncons.rows(), 0, Ncons.cols()), [&] (blocked_range2d<size_t>& r)
            {
                for (auto i = r.cols().begin(); i < r.cols().end(); i++)
                {
                    if (i == r.cols().begin())
                    {
                        for (auto k = 0; k < dom_dim; k++)
                        {
                            thread_local_knot_idxs.local()[k].resize(mfa_data.p(k) + 2);
                            thread_local_knots.local()[k].resize(mfa_data.p(k) + 2);
                        }
                    }

                    // local knot vector
                    mfa_data.tmesh.knot_intersections(anchors[i], t_idx_anchors[i], thread_local_knot_idxs.local());
                    for (auto k = 0; k < dom_dim; k++)
                    {
                        for (auto n = 0; n < thread_local_knot_idxs.local()[k].size(); n++)
                            thread_local_knots.local()[k][n] = mfa_data.tmesh.all_knots[k][thread_local_knot_idxs.local()[k][n]];
                    }

                    for (auto j = r.rows().begin(); j < r.rows().end(); j++)
                    {
                        dom_iter.idx_ijk(j, thread_dom_ijk.local());                        // ijk of domain point
                        for (auto k = 0; k < dom_dim; k++)
                        {
                            T u = input.params->param_grid[k][thread_dom_ijk.local()(k)];   // parameter of current input point
                            T B = mfa_data.OneBasisFun(k, u, thread_local_knots.local()[k]);// basis function
                            if (Ncons(j, i) == -1.0)                                        // unassigned so far
                                Ncons(j, i) = B;
                            else
                                Ncons(j, i) *= B;
                        }
                    }       // for blocked range rows
                }       // for blocked range cols
            }, ap); // parallel for

#else       // serial

            for (auto i = 0; i < Ncons.cols(); i++)                                             // for all constraint control points
            {
                // local knot vector
                mfa_data.tmesh.knot_intersections(anchors[i], t_idx_anchors[i], local_knot_idxs);
                for (auto k = 0; k < dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                // iterator over input points
                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());
                while (!dom_iter.done())
                {
                    for (auto k = 0; k < dom_dim; k++)                                 // for all dims
                    {
                        T u = input.params->param_grid[k][dom_iter.idx_dim(k)];                             // parameter of current input point
                        T B = mfa_data.OneBasisFun(k, u, local_knots[k]);                       // basis function
                        if (Ncons(dom_iter.cur_iter(), i) == -1.0)                              // unassigned so far
                            Ncons(dom_iter.cur_iter(), i) = B;
                        else
                            Ncons(dom_iter.cur_iter(), i) *= B;
                    }           // for all dims
                    dom_iter.incr_iter();
                }           // domain points iterator
            }       // for all constraint control points

#endif      // TBB or serial

            // set any unassigned values in Ncons to 0
            for (auto i = 0; i < Ncons.rows(); i++)
            {
                for (auto j = 0; j < Ncons.cols(); j++)
                    if (Ncons(i, j) == -1.0)
                        Ncons(i, j) = 0.0;
            }
        }

        // encodes the control points for one tensor product of a tmesh
        // takes a subset of input points from the global domain, covered by basis functions of this tensor product
        // solves all dimensions together (not separably)
        // does not encode weights for now
        // latest linear constrained formulation as proposed by David Lenz (see wiki/notes/linear-constrained-fit.pdf)
        void EncodeTensorLocalUnified(
                TensorIdx                 t_idx,                  // index of tensor product being encoded
                bool                      weighted = true)        // solve for and use weights
        {
            // additional constraints
            // it turns out these don't change anything, leave set to 0
            int extra_cons = 0;

            // timing
            double setup_time   = MPI_Wtime();
            double q_time       = MPI_Wtime();
            fmt::print(stderr, "\nSetting up...\n");

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];                               // current tensor product
            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                                   // control point dimensionality

            // get input domain points covered by the tensor and the constraints
            vector<size_t> start_idxs(mfa_data.dom_dim);
            vector<size_t> end_idxs(mfa_data.dom_dim);
            mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, true, extra_cons, start_idxs, end_idxs);

            // Q matrix of relevant input domain points
            VectorXi ndom_pts(dom_dim);
            VectorXi dom_starts(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];                                                    // need Eigen vector from STL vector
            }
            MatrixX<T> Q(ndom_pts.prod(), pt_dim);
            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());
            while (!dom_iter.done())
            {
                Q.block(dom_iter.cur_iter(), 0, 1, pt_dim) =
                    input.domain.block(dom_iter.sub_full_idx(dom_iter.cur_iter()), mfa_data.min_dim, 1, pt_dim);
                dom_iter.incr_iter();
            }

            // resize control points and weights in case number of control points changed
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());
            t.weights = VectorX<T>::Ones(t.weights.size());                                         // linear solve does not solve for weights; set to 1

            // timing
            q_time                  = MPI_Wtime() - q_time;
            double free_time        = MPI_Wtime();

            // matrix of free control point basis functions

            // a dense matrix copied to a sparse matrix just prior to solving
            // To fill sparse matrix directly, skipping dense matrix, we would need to use row-major order
            // to do row-sum normalization efficiently later. However, row-major is 2X slower to fill
            // and 3-4X slower to solve than column-major. Hence, for now, we fill a dense matrix,
            // do the row normalization, then copy dense matrix to column-major sparse matrix just before solving.

            MatrixX<T> Nfree;
            int max_nnz_col = FreeCtrlPtMat(t_idx, ndom_pts, dom_starts, Nfree);                    // returns max num nonzeros in any column

            // timing
            free_time           = MPI_Wtime() - free_time;
            double cons_time    = MPI_Wtime();

            // find constraint control points and their anchors
            MatrixX<T>                  Pcons;                                                      // constraint control points
            vector<vector<KnotIdx>>     anchors;                                                    // corresponding anchors
            vector<TensorIdx>           t_idx_anchors;                                              // tensors containing corresponding anchors

#ifndef MFA_NO_CONSTRAINTS      // for debugging can disable constraints; normally not used

            LocalSolveAllConstraints(t, Pcons, anchors, t_idx_anchors);

#endif

            MatrixX<T> Ncons = MatrixX<T>::Constant(Q.rows(), Pcons.rows(), -1);                    // basis functions, -1 means unassigned so far
            if (Pcons.rows())
                ConsCtrlPtMat(ndom_pts, dom_starts, anchors, t_idx_anchors, Ncons);

            // timing
            cons_time           = MPI_Wtime() - cons_time;

            // normalize Nfree and Ncons such that the row sum of Nfree + Ncons = 1.0
            double norm_time    = MPI_Wtime();              // timing

#ifdef MFA_TBB

            static affinity_partitioner ap;
            parallel_for (blocked_range<size_t>(0, Nfree.rows()), [&] (blocked_range<size_t>& r)
            {
            for (auto i = r.begin(); i < r.end(); i++)

#else

            for (auto i = 0; i < Nfree.rows(); i++)

#endif

            {
                bool error = false;
                T sum = Nfree.row(i).sum();
                if (Pcons.rows())
                    sum += Ncons.row(i).sum();

                if (sum > 0.0)
                {
                    Nfree.row(i) /= sum;
                    if (Pcons.rows())
                        Ncons.row(i) /= sum;
                }
                else
                {
                    if (Pcons.rows())
                        fmt::print(stderr, "Warning: EncodeTensorLocalUnified(): row {} Nfree row sum = {} Ncons row sum = {}, Nfree + Ncons row sum = {}. This should not happen.\n",
                            i, Nfree.row(i).sum(), Ncons.row(i).sum(), sum);
                    else
                        fmt::print(stderr, "Warning: EncodeTensorLocalUnified(): row {} Nfree row sum = {}. This should not happen.\n",
                            i, sum);
                    error = true;
                }
                if (error)
                {
                    VectorXi ijk(dom_dim);
                    dom_iter.idx_ijk(i, ijk);
                    cerr << "ijk = " << ijk.transpose() << endl;
                    fmt::print(stderr, "params = [ ");
                    for (auto k = 0; k < dom_dim; k++)
                        fmt::print(stderr, "{} ", input.params->param_grid[k][ijk(k)]);
                    fmt::print(stderr, "]\n");
                }
            }

#ifdef MFA_TBB

            }, ap);

#endif

            norm_time = MPI_Wtime() - norm_time;                // timing

            // multiply by transpose to make the matrix square and smaller
            double mult_time    = MPI_Wtime();                  // timing
            MatrixX<T> NtNfree  = Nfree.transpose() * Nfree;
            mult_time           = MPI_Wtime() - mult_time;      // timing

#ifndef MFA_DENSE

            // compute max nonzero columns in NtNfree
            max_nnz_col = 0;
            for (auto i = 0; i < NtNfree.rows(); i++)
            {
                int nnz_col = 0;
                for (auto j = 0; j < NtNfree.cols(); j++)
                {
                    if (NtNfree(i, j) != 0.0)
                        nnz_col++;
                }
                if (nnz_col > max_nnz_col)
                    max_nnz_col = nnz_col;
            }

            // copy from dense to sparse
            SparseMatrixX<T> NtNfree_sparse(NtNfree.rows(), NtNfree.cols());
            NtNfree_sparse.reserve(VectorXi::Constant(NtNfree.cols(), max_nnz_col));
            for (auto i = 0; i < NtNfree.rows(); i++)
            {
                for (auto j = 0; j < NtNfree.cols(); j++)
                {
                    if (NtNfree(i, j) != 0.0)
                        NtNfree_sparse.insert(i, j) = NtNfree(i,j);
                }
            }
            NtNfree_sparse.makeCompressed();

#endif

            // timing
            double r_time       = MPI_Wtime();

            // R is the right hand side needed for solving N * P = R
            MatrixX<T> R = Q;
            if (Pcons.rows())
                R -= Ncons * Pcons;

            // timing
            r_time                  = MPI_Wtime() - r_time;
            setup_time              = MPI_Wtime() - setup_time;

            fmt::print(stderr, "Solving...\n");

#ifdef MFA_DENSE    // dense solve

            double dense_solve_time = MPI_Wtime();                  // timing
            t.ctrl_pts = (Nfree.transpose() * Nfree).ldlt().solve(Nfree.transpose() * R);
            dense_solve_time = MPI_Wtime() - dense_solve_time;

#else               // sparse solve

            double sparse_solve_time    = MPI_Wtime();              // timing

            Eigen::SparseQR<SparseMatrixX<T>, Eigen::COLAMDOrdering<int>>  solver;

            // TODO: iterative least squares conjugate gradient is faster but sometimes fails
            // NtN not necessarily symmetric positive definite?

            solver.compute(NtNfree_sparse);

            if (solver.info() != Eigen::Success)
            {
                cerr << "EncodeTensorLocalUnified(): Error: Matrix decomposition failed" << endl;
                abort();
            }

            t.ctrl_pts = solver.solve(Nfree.transpose() * R);
            if (solver.info() != Eigen::Success)
            {
                cerr << "EncodeTensorLocalUnified(): Error: Least-squares solve failed" << endl;
                abort();
            }

            sparse_solve_time = MPI_Wtime() - sparse_solve_time;

#endif

            // timing
            fmt::print(stderr, "EncodeTensorLocalUnified() timing:\n");
            fmt::print(stderr, "setup time: {} s.\n", setup_time);
            fmt::print(stderr, "    = free time {} + cons time {} + mult time {} s.\n",
                    free_time, cons_time, mult_time);

#ifdef MFA_DENSE
            fmt::print(stderr, "dense_solve time: {} s.\n", dense_solve_time);
#else
            fmt::print(stderr, "sparse_solve time: {} s.\n", sparse_solve_time);
#endif

            // Report the relative residual of the solve.
            double relative_error = (Nfree * t.ctrl_pts - R).norm() / R.norm(); // norm() is L2 norm
            cerr << "EncodeTensorLocalUnified(): The relative error is " << relative_error << endl;
        }

        // encodes the control points for one tensor product of a tmesh
        // takes a subset of input points from the global domain, covered by basis functions of this tensor product
        // solves dimensions separably
        // does not encode weights for now
        // linear constrained formulation as proposed by David Lenz (see wiki/notes/linear-constrained-fit.pdf)
        void EncodeTensorLocalSeparable(
                TensorIdx                 t_idx,                  // index of tensor product being encoded
                bool                      weighted = true)        // solve for and use weights
        {
            // additional constraints
            // it turns out these don't change anything, leave set to 0
            int extra_cons = 0;

            double t0 = MPI_Wtime();

            // typing shortcuts
            auto& tmesh         = mfa_data.tmesh;
            auto& tensor_prods  = tmesh.tensor_prods;

            auto& t = tensor_prods[t_idx];                                                          // current tensor product
            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                                   // control point dimensionality

            // get input domain points covered by the tensor, including constraints
            vector<size_t> start_idxs(dom_dim);                                                     // start of input points including constraints
            vector<size_t> end_idxs(dom_dim);                                                       // end of input points including constraints
            mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, false, extra_cons, start_idxs, end_idxs); // false: do not extend to cover constraints

            // relevant input domain points covering constraints
            VectorXi ndom_pts(dom_dim);
            VectorXi dom_starts(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];                                                    // need Eigen vector from STL vector
            }

            // resize control points and weights in case number of control points changed
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());
            t.weights = VectorX<T>::Ones(t.weights.size());                                         // linear solve does not solve for weights; set to 1

            // max size of input point matrices is the product of the max of input points and control points in each dim
            size_t max_npts = 1;
            for (auto k = 0; k < dom_dim; k++)
                max_npts *= std::max(ndom_pts(k), t.nctrl_pts(k));

            // two matrices of input points for subsequent dimensions after dimension 0, which draws directly from input domain
            // input points cover constraints as well as free control point basis functions
            // (double buffering output control points to input points)
            MatrixX<T> Q(max_npts, pt_dim);
            MatrixX<T> Q1(max_npts, pt_dim);

            // number of output points in current dimension and number of input points of next dimension
            VectorXi npts = ndom_pts;
            npts(0) = t.nctrl_pts(0);

            // input and output number of points
            VectorXi nin_pts    = ndom_pts;
            VectorXi nout_pts   = npts;
            VectorXi in_starts  = dom_starts;
            VectorXi in_all_pts = input.ndom_pts();
            VectorXi start_ijk  = in_starts;

            // timing
            double free_time    = 0.0;
            double cons_time    = 0.0;
            double norm_time    = 0.0;
            double solve_time   = 0.0;

            for (auto dim = 0; dim < dom_dim; dim++)                                                // for all domain dimensions
            {
                MatrixX<T> R(nin_pts(dim), pt_dim);                                                 // RHS for solving N * P = R

                VolIterator     in_iter(nin_pts, in_starts, in_all_pts);                            // volume of current input points
                VolIterator     out_iter(nout_pts);                                                 // volume of current output points
                SliceIterator   in_slice_iter(in_iter, dim);                                        // slice of the input points volume missing current dim
                SliceIterator   out_slice_iter(out_iter, dim);                                      // slice of the output points volume missing current dim

                // allocate matrices of free and constraint control points and constraint basis functions
                MatrixX<T>  Nfree(nin_pts(dim), t.nctrl_pts(dim));
                ConsType    cons_type;                                                              // none, left, right, both
                MatrixX<T>  Ncons;
                MatrixX<T>  Pcons;
                MatrixX<T>  P(t.nctrl_pts(dim), pt_dim);

#ifdef MFA_NO_CONSTRAINTS

                cons_type = ConsType::MFA_NO_CONSTRAINT;
                int ncons = 0;

#else       // MFA_NO_CONSTRAINTS

                int ncons = mfa_data.p(dim) + extra_cons;

                if (t.knot_mins[dim] == 0 && t.knot_maxs[dim] == tmesh.all_knots[dim].size() - 1)
                    cons_type = ConsType::MFA_NO_CONSTRAINT;

                vector<TensorIdx>       t_idx_anchors;                                                  // tensors containing corresponding anchors
                vector<vector<KnotIdx>> anchors;                                                        // corresponding anchors
                VectorXi                ncurve_pts = ndom_pts;

                if (dim > 0)
                    cons_type = ConsType::MFA_NO_CONSTRAINT;
                else if (dim == 0 && t.knot_mins[dim] == 0)
                    cons_type   = ConsType::MFA_RIGHT_ONLY_CONSTRAINT;
                else if (dim == 0 && t.knot_maxs[dim] == tmesh.all_knots[dim].size() - 1)
                    cons_type   = ConsType::MFA_LEFT_ONLY_CONSTRAINT;
                else if (dim == 0)
                    cons_type   = ConsType::MFA_BOTH_CONSTRAINT;

                if (dim == 0)
                {
                    // fill Pcons
                    LocalSolveAllConstraints(t, Pcons, anchors, t_idx_anchors);

                    // number of input points in one curve
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        if (j != dim)
                            ncurve_pts[j] = 1;
                    }
                }

#endif      // MFA_NO_CONSTRAINTS

                // for all curves in the current dimension
                while (!in_slice_iter.done())                                                       // for all curves
                {

                    CurveIterator   in_curve_iter(in_slice_iter);                                   // one curve of the input points in the current dim
                    start_ijk = in_curve_iter.cur_ijk();
                    // add original input point starting offsets to higher dims start_ijk
                    // but don't offset the previous dims: start_ijk for previous dims refers to control points, not input points
                    if (dim > 0)
                    {
                        for (auto j = dim; j < dom_dim; j++)
                            start_ijk(j) += dom_starts(j);
                    }

                    if (CurveIntersectsTensor(t_idx, dim, nin_pts, start_ijk))
                    {
                        // fill Ncons
                        if (dim == 0 && Pcons.rows())
                        {
                            Ncons = MatrixX<T>::Constant(ncurve_pts[0], Pcons.rows(), -1);                    // basis functions, -1 means unassigned so far
                            ConsCtrlPtMat(ncurve_pts, start_ijk, anchors, t_idx_anchors, Ncons);
                        }

                        ComputeCtrlPtCurve(in_curve_iter, t_idx, dim, R, Q, Q1, Nfree, Ncons, Pcons, P,
                                cons_type, nin_pts, start_ijk, ncons, free_time, cons_time, norm_time, solve_time);
                    }

                    // copy solution to one curve of output points
                    CurveIterator   out_curve_iter(out_slice_iter);                                 // one curve of the output points in the current dim
                    while (!out_curve_iter.done())
                    {
                        if (dim % 2 == 0)
                            Q1.row(out_curve_iter.ijk_idx(out_curve_iter.cur_ijk())) = P.row(out_curve_iter.cur_iter());
                        else
                            Q.row(out_curve_iter.ijk_idx(out_curve_iter.cur_ijk())) = P.row(out_curve_iter.cur_iter());
                        out_curve_iter.incr_iter();
                    }
                    out_slice_iter.incr_iter();
                    in_slice_iter.incr_iter();
                }       // for all curves in current dim

                // adjust input, output numbers of points for next iteration
                nin_pts(dim) = t.nctrl_pts(dim);
                if (dim < dom_dim - 1)
                    nout_pts(dim + 1)   = t.nctrl_pts(dim + 1);
                in_starts   = VectorXi::Zero(dom_dim);                  // subvolume = full volume for later dims
                in_all_pts  = nin_pts;
            }       // for all domain dimensions

            // copy final result back to tensor product
            if (dom_dim % 2 == 0)
                t.ctrl_pts = Q.block(0, 0, t.nctrl_pts.prod(), pt_dim);
            else
                t.ctrl_pts = Q1.block(0, 0, t.nctrl_pts.prod(), pt_dim);

            // timing
            fmt::print(stderr, "EncodeTensorLocalSeparable() tidx {} time {:.3e} s.\n", t_idx, MPI_Wtime() - t0);
        }

#endif  // MFA_TMESH

#ifdef MFA_LOW_D

        // original adaptive encoding for first tensor product only
        // older version using 1-d curves for new knots
        void OrigAdaptiveEncode(
                T                   err_limit,              // maximum allowable normalized error
                bool                weighted,               // solve for and use weights
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)         // optional maximum number of rounds
        {
            vector<vector<T>> new_knots;                               // new knots in each dim.

            {
                fmt::print(stderr, "Using OrigAdaptiveEncode() w/ 1-d curve knot splitting\n");
            }
            if (verbose >= 3)
            {
#ifdef MFA_CHECK_ALL_CURVES
                fmt::print(stderr, "Checking all curves (slower but more accurate)\n");
#else
                fmt::print(stderr, "Checking a sampling of curves (faster but less accurate)\n");
#endif
            }


            // TODO: use weights for knot insertion
            // for now, weights are only used for final full encode

            // loop until no change in knots
            for (int iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                {
                        fprintf(stderr, "\nDone; max iterations reached.\n\n");
                    break;
                }

                    fprintf(stderr, "--- Iteration %d ---\n", iter);

                // low-d w/ splitting spans in the middle
                bool done = OrigNewKnots_curve(new_knots, err_limit, extents, iter);

                // no new knots to be added
                if (done)
                {
                        fprintf(stderr, "\nDone; no new knots added.\n\n");
                    break;
                }

                // check if the new knots would make the number of control points >= number of input points in any dim
                done = false;
                for (auto k = 0; k < dom_dim; k++)
                    // hard-coded for first tensor
                    if (input.ndom_pts(k) <= mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) + new_knots[k].size())
                    {
                        done = true;
                        break;
                    }
                if (done)
                {
                        fprintf(stderr, "\nDone; control points would outnumber input points.\n\n");
                    break;
                }

            }

            // final full encoding needed after last knot insertion above
                fmt::print("Performing final encoding\n");
            TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];        // fixed encode assumes the tmesh has only one tensor product
            Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);
        }

#else       // full-dimensional knot insertion

        // original adaptive encoding for first tensor product only
        // latest version using full knot spans (5/19/21)
        void OrigAdaptiveEncode(
                T                   err_limit,              // maximum allowable normalized error
                bool                weighted,               // solve for and use weights
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)         // optional maximum number of rounds
        {
            vector<vector<T>> new_knots;                               // new knots in each dim.
            ErrorStats<T> error_stats;

            fmt::print(stderr, "Using OrigAdaptiveEncode() w/ full-d knot splitting\n\n");

            VectorX<T> myextents = extents.size() ? extents : VectorX<T>::Ones(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());

            mfa::NewKnots<T> nk(mfa_data, input);

            // loop until no change in knots
            for (int iter = 0; ; iter++)
            {
                // encode tensor product 0
                TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];
                for (auto j = 0; j < dom_dim; j++)
                    t.nctrl_pts[j] = mfa_data.tmesh.all_knots[j].size() - mfa_data.p(j) - 1;
                Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);

                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; max iterations reached.\n\n");
                    break;
                }

                if (verbose)
                    fprintf(stderr, "\n--- Iteration %d ---\n", iter);

                // check all knots spans for error
                vector<vector<KnotIdx>>     inserted_knot_idxs(dom_dim);   // indices in each dim. of inserted knots in full knot vector after insertion
                vector<vector<T>>           inserted_knots(dom_dim);       // knots to be inserted in each dim.
                vector<TensorIdx>           parent_tensor_idxs;                     // tensors having knots inserted
                vector<TensorProduct<T>>    new_tensors;                            // empty, unused
                bool done = nk.AllErrorSpans(
                        0,
                        myextents,
                        err_limit,
                        true,
                        new_tensors,
                        parent_tensor_idxs,
                        inserted_knot_idxs,
                        inserted_knots,
                        error_stats);

                // no new knots to be added
                if (done)
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; no new knots added.\n\n");
                    break;
                }

                // insert new knots into knot vectors
                int n_insertions = parent_tensor_idxs.size();                       // number of knots to insert
                for (auto j = 0; j < dom_dim; j++)
                    assert(inserted_knot_idxs[j].size() == n_insertions &&
                            inserted_knots[j].size() == n_insertions);

                vector<bool> inserted(dom_dim);                            // whether the current insertion succeed (in each dim)

                for (auto i = 0; i < n_insertions; i++)                             // for all knots to be inserted
                {
                    // insert the new knots into tmesh all_knots
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        inserted[j] = false;
                        if (mfa_data.tmesh.insert_knot(
                                    j,
                                    0,                                              // all knots at level 0 in this version
                                    inserted_knots[j][i],
                                    input.params->param_grid,
                                    inserted_knot_idxs[j][i]))
                        {
                            inserted[j] = true;
                            // increment subsequent insertions
                            for (auto k = 0; k < n_insertions; k++)
                            {
                                if (inserted_knot_idxs[j][k] > inserted_knot_idxs[j][i])
                                    inserted_knot_idxs[j][k]++;
                            }
                        }
                    }   // dimension
                }   // knot insertions

                if (verbose)
                    PrintAdaptiveStats(error_stats);
            }   // iterations
        }

#endif

        // adaptive encoding for T-mesh
        void AdaptiveEncode(
                T                   err_limit,                          // maximum allowable normalized error
                bool                weighted,                           // solve for and use weights
                const VectorX<T>&   extents,                            // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)                     // optional maximum number of rounds
        {
            auto& tmesh         = mfa_data.tmesh;
            auto& tensor_prods  = tmesh.tensor_prods;
            int parent_level = 0;                                       // parent level currently being refined

            // timing
            double t0 = MPI_Wtime();

            TensorProduct<T>&t = tensor_prods[0];                       // fixed encode assumes the tmesh has only one tensor product
            Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);

            // timing
            fmt::print(stderr, "\nInitial full encode time:   {:.3e} s.\n", MPI_Wtime() - t0);

            vector<TensorProduct<T>>    new_tensors;                    // newly refined tensors to be added

            // intersection proximity between tensors (assumes same for all dims)
            VectorXi& p = mfa_data.p;

            // padding for all tensors
            // following was found empirically to be best: even p: pad = p + 1; odd p: pad = p + 2
            // only tested p=2 and p=3, however
            int pad         = p(0) % 2 == 0 ? p(0) + 1 : p(0) + 2;

            // tightest pad possible, leaving only p control points in a tensor
            int tightest_pad = (p(0) % 2 == 0 ? p(0) : p(0) - 1);

            // extra padding for tensor at global edge
            int edge_pad    = (p(0) / 2) + 2;

            // loop until all tensors are done
            int iter;
            for (iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                    break;

                if (verbose)
                    fmt::print(stderr, "\n--- Iteration {}, refining level {} ---\n", iter, parent_level);


                Refine(parent_level, iter + 1, err_limit, extents, pad, edge_pad, new_tensors);

                UpdateExistingTensors(new_tensors);

                // check if number of control points exceeds input points in any dimension (in the case of a single tensor)
                if (tensor_prods.size() == 1)
                {
                    bool done = false;
                    int k;
                    for (k = 0; k < mfa_data.dom_dim; k++)
                    {
                        if (input.ndom_pts(k) <= tensor_prods[0].nctrl_pts(k))
                        {
                            done = true;
                            break;
                        }
                    }
                    if (done)
                    {
                        if (verbose)
                            fmt::print(stderr, "\nKnot insertion done after {} iterations; Control points in dim. {} is {} which is the max.\n\n",
                                    iter + 1, k, tensor_prods[0].nctrl_pts(k));
                        break;
                    }
                }
                // check if all levels are done
                if (parent_level >= tmesh.max_level)
                {
                    if (verbose)
                        fmt::print(stderr, "\nKnot insertion done after {} iterations; no new knots added.\n\n", iter + 1);
                    break;
                }
                else                                                // one iteration only is done
                {
                    CheckNewTensors(new_tensors, pad, edge_pad);
                    fmt::print(stderr, "Level {} done, adding {} new tensor(s)\n", parent_level, new_tensors.size());
                    double add_tensors_time = MPI_Wtime();
                    bool add_success = AddNewTensors(new_tensors, tightest_pad);
                    add_tensors_time = MPI_Wtime() - add_tensors_time;
                    fmt::print(stderr, "Solving and adding new tensors time:   {:.3e} s.\n", add_tensors_time);
                    if (add_success)
                        parent_level++;
                    new_tensors.clear();
                }
            }   // iterations

            fmt::print(stderr, "{} iterations.\n", iter + 1);
            fmt::print(stderr, "{} tensor products.\n", tensor_prods.size());

            // Verify that every input point is covered by a knot span.
            mfa::NewKnots<T> nk(mfa_data, input);
            if (!nk.CheckAllSpans())
                throw MFAError(fmt::format("AdaptiveEncode(): Error: failed checking all spans for input points\n"));

            // Report tensors left unfinished by a capped refinement run.
            bool all_done = true;
            for (auto i = 0; i < tensor_prods.size(); i++)
            {
                bool first = true;
                auto& t = tensor_prods[i];
                if (!t.done)
                {
                    if (first)
                        fmt::print(stderr,"\n");
                    fmt::print(stderr, "Tensor {} level {} is not marked done.\n", i, t.level);
                    fmt::print(stderr, "This is normal if the number of rounds is capped; otherwise this should not happen.\n");
                    all_done    = false;
                    first       = false;
                }
            }
            if (all_done)
                fmt::print(stderr, "All tensors are marked as done.\n");

            // Enforce the minimum control-point count for every tensor.
            for (auto i = 0; i < tensor_prods.size(); i++)
            {
                if (!tmesh.check_num_ctrl_degree(i, 0))
                {
                    auto& t = tensor_prods[i];
                    fmt::print(stderr, "Error: Refine(): tensor {} has fewer than p control points. This should not happen\n", i);
                    fmt::print(stderr, "knot_mins [{}] knot_maxs[{}] level {}\n",
                            fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level);
                    fmt::print(stderr, "\nAddNewTensors(): T-mesh:\n");
                    tmesh.print(true, true, false, false);
                    abort();
                }
            }

            // Verify that tensor-local knots agree with the global knot vectors.
            if (!tmesh.check_local_knots())
                throw MFAError(fmt::format("AdaptiveEncode(): Error: failed checking local knots of tensors\n"));

        }

    private:

        // print max error, compression factor, and any pother stats at the end of an adaptive iteration
        void PrintAdaptiveStats(
                const ErrorStats<T>&    error_stats)
        {
            // compute compression
            float in_coords = (input.npts) * (input.pt_dim);
            float out_coords = 0.0;
            for (auto i = 0; i < mfa_data.tmesh.tensor_prods.size(); i++)
                out_coords += mfa_data.tmesh.tensor_prods[i].ctrl_pts.rows() *
                    mfa_data.tmesh.tensor_prods[i].ctrl_pts.cols();
            for (auto j = 0; j < mfa_data.tmesh.all_knots.size(); j++)
                out_coords += mfa_data.tmesh.all_knots[j].size();
            float compression = in_coords / out_coords;

            T rms_abs_err = sqrt(error_stats.sum_sq_abs_errs / (input.domain.rows()));
            T rms_norm_err = sqrt(error_stats.sum_sq_norm_errs / (input.domain.rows()));

            fprintf(stderr, "\n----- estimates of current variable of current model -----\n");
            fprintf(stderr, "estimated max_err               = %e\n",  error_stats.max_abs_err);
            fprintf(stderr, "estimated normalized max_err    = %e\n",  error_stats.max_norm_err);
            fprintf(stderr, "estimated RMS error             = %e\n",  rms_abs_err);
            fprintf(stderr, "estimated normalized RMS error  = %e\n",  rms_norm_err);
            fprintf(stderr, "estimated compression ratio     = %.2f\n",  compression);
            fprintf(stderr, "-----------------------------------------------------------\n");
        }

#ifndef      MFA_NO_WEIGHTS

        bool Weights(
                int                 k,              // current dimension
                const MatrixX<T>&   Q,              // input points
                const MatrixX<T>&   N,              // basis functions
                const MatrixX<T>&   NtN,            // N^T * N
                int                 curve_id,       // debugging
                VectorX<T>&         weights)        // (output) weights
        {
            bool success;

            // Nt, NtNi
            // TODO: offer option of saving time or space by comuting Nt and NtN each time it is needed?
            MatrixX<T> Nt   = N.transpose();
            MatrixX<T> NtNi = NtN.partialPivLu().inverse();

            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                     // dimensionality of input and control points (domain and range)

            MatrixX<T> NtQ2  = MatrixX<T>::Zero(Nt.rows(),   Nt.cols());    // N^T * Q^2
            MatrixX<T> NtQ   = MatrixX<T>::Zero(Nt.rows(),   Nt.cols());    // N^T * Q
            MatrixX<T> NtQ2N = MatrixX<T>::Zero(NtN.rows(),  NtN.cols());   // N^T * Q^2 * N
            MatrixX<T> NtQN  = MatrixX<T>::Zero(NtN.rows(),  NtN.cols());   // N^T * Q   * N

            // temporary matrices NtQ and NtQ2
            for (auto i = 0; i < Nt.cols(); i++)
            {
                NtQ.col(i)  = Nt.col(i) * Q(i, pt_dim - 1);
                NtQ2.col(i) = Nt.col(i) * Q(i, pt_dim - 1) * Q(i, pt_dim - 1);
            }

            // final matrices NtQN and NtQ2N
            NtQN  = NtQ  * N;
            NtQ2N = NtQ2 * N;

            // compute the matrix M according to eq.3 and eq. 4 of M&K95
            MatrixX<T> M = NtQ2N - NtQN * NtNi * NtQN;

            // compute the eigenvalues and eigenvectors of M (eq. 9 of M&K95)
            // NB: M is self-adjoint (ie, real and symmetric)
            // Eigen's SelfAdjointEigenSolver is faster than its normal EigenSolver in this case
            Eigen::SelfAdjointEigenSolver<MatrixX<T>> eigensolver(M);
            if (eigensolver.info() != Eigen::Success)
            {
                fprintf(stderr, "Error: Encoder::Weights(), computing eigenvalues of M failed, perhaps M is not self-adjoint?\n");
                return false;
            }

            const MatrixX<T>& EV    = eigensolver.eigenvectors();          // typing shortcut
            const VectorX<T>& evals = eigensolver.eigenvalues();           // typing shortcut

            // eigenvalues should be positive and distinct
            for (auto i = 0; i < evals.size() - 1; i++)
                if (evals(i) == 0.0 || evals(i) == evals(i + 1))
                {
                    fprintf(stderr, "Warning: Weights(): eigenvalues should be positive and distinct.\n");
                    fprintf(stderr, "Aborting weights calculation\n");
                    return false;
                }

            // if smallest eigenvector is all positive or all negative, those are the weights
            if ( (EV.col(0).array() > 0.0).all() )
            {
                weights = EV.col(0);
                weights *= (1.0 / weights.maxCoeff());  // scale to max weight = 1
                success = true;
            }
            else if ( (EV.col(0).array() < 0.0).all() )
            {
                weights = -EV.col(0);
                weights *= (1.0 / weights.maxCoeff());  // scale to max weight = 1
                success = true;
            }

            // if smallest eigenvector is mixed sign, then expand eigen space
            else
            {
                success      = false;
                T min_weight = 1.0;
                T max_weight = 1.0e4;
                // minimum eigenvector element, if less, clamp to 0.0 (between 1e-6 and 1e-12 seems to help)
                T min_ev_val = 1.0e-12;

                auto nweights = weights.size();

                // column indices
                vector<int> col_idx(2 * nweights);
                for (auto j = 0; j < 2 * nweights; j++)
                    col_idx[j] = j;

                ClpSimplex model;                           // simplex method, ~10X faster than interior point
                model.resize(2 * nweights, 0);              // all the rows and no columns (columns are added incrementally)

                // row upper bounds
                for (auto i = 0; i < nweights; i++)
                {
                    model.setRowUpper(i, max_weight);
                    model.setRowUpper(nweights + i, -min_weight);
                }

                // first column of the matrix
                vector<double> newcol(nweights * 2);
                for (auto j = 0; j < nweights; j++)
                {
                    newcol[j]            = fabs(EV(j, 0)) < min_ev_val ? 0.0 : EV(j, 0);
                    newcol[nweights + j] = -newcol[j];
                }
                model.addColumn(2 * nweights, &col_idx[0], &newcol[0], -COIN_DBL_MAX, COIN_DBL_MAX);

                for (auto i = 2; i <= EV.cols(); i++)        // expand from 2 eigenvectors to all, one at a time
                {
                    // add another column
                    for (auto j = 0; j < nweights; j++)
                    {
                        newcol[j]            = fabs(EV(j, i - 1)) < min_ev_val ? 0.0 : EV(j, i - 1);
                        newcol[nweights + j] = -newcol[j];
                    }
                    model.addColumn(2 * nweights, &col_idx[0], &newcol[0], -COIN_DBL_MAX, COIN_DBL_MAX);

                    // solve
                    model.setLogLevel(0);
                    model.initialSolve();               // simplex method, ~10X faster than interior point

                    if (!model.isProvenPrimalInfeasible() && !model.isIterationLimitReached())
                    {
                        // copy out the solution
                        VectorX<T>    solved_weights = VectorX<T>::Zero(nweights);
                        int           ncols          = model.getNumCols();
                        const double* colSol         = model.getColSolution();
                        for (auto k = 0; k < ncols; k++)
                            solved_weights += colSol[k] * EV.col(k);

                        // check if the solution was found successfully
                        if ( (solved_weights.array() > 0.0).all() )
                        {
                            weights = solved_weights;
                            weights *= (1.0 / weights.maxCoeff());  // scale to max weight = 1
                            success = true;
                            if (verbose)
                                fprintf(stderr, "curve %d: successful linear solve from %d eigenvectors\n", curve_id, i);
                        }
                    }

                    if (success)
                        break;
                }                                               // increasing number of eigenvectors

                if (!success)
                {
                    weights = VectorX<T>::Ones(nweights);
                    if (verbose)
                        fprintf(stderr, "curve %d: linear solver could not find positive weights; setting to 1\n", curve_id);
                }
            }                                                   // else need to expand eigenspace

            return success;
        }

#endif

        // computes right hand side vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve from the
        // original input domain points
        // includes multiplication by weights
        // R is column vector of n + 1 elements, each element multiple coordinates of the input points
        void RHS(
                int                 cur_dim,  // current dimension
                const MatrixX<T>&   N,        // matrix of basis function coefficients
                MatrixX<T>&         R,        // (output) residual matrix allocated by caller
                const VectorX<T>&   weights,  // precomputed weights for n + 1 control points on this curve
                int                 co)       // index of starting domain pt in current curve
        {
            int last   = R.cols() - 1;                                          // column of range value TODO: weighing only the last column does not make much sense in the split model
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);   // one row for each input point
            VectorX<T> denom(N.rows());                                         // rational denomoninator for param of each input point

            for (int k = 0; k < N.rows(); k++)                                  // for all input points
            {
                denom(k) = (N.row(k).cwiseProduct(weights.transpose())).sum();
#ifdef UNCLAMPED_KNOTS
                if (denom(k) == 0.0)
                    denom(k) = 1.0;
#endif
                Rk.row(k) = input.domain.block(co + k * input.g.ds[cur_dim], mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
            }

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                                     // ith basis functions for input pts
                         weights(i) / denom.array() *                           // rationalized
                         Rk.col(j).array()).sum();                              // input points
#else                                                                           // don't weigh domain coordinate (only range)

            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
            {
                // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                for (int j = 0; j < R.cols() - 1; j++)
                    R(i, j) =
                        (N.col(i).array() *                                     // ith basis functions for input pts
                         Rk.col(j).array()).sum();                              // input points
                R(i, last) =
                    (N.col(i).array() *                                         // ith basis functions for input pts
                     weights(i) / denom.array() *                               // rationalized
                     Rk.col(last).array()).sum();                               // input points
            }
#endif

        }


        // computes right hand side vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve from a
        // new set of input points, not the default input domain
        // includes multiplication by weights
        // R is column vector of n + 1 elements, each element multiple coordinates of the input points
        void RHS(
                int                 cur_dim,  // current dimension
                const MatrixX<T>&   in_pts,   // input points (not the default domain stored in the mfa)
                const MatrixX<T>&   N,        // matrix of basis function coefficients
                MatrixX<T>&         R,        // (output) residual matrix allocated by caller
                const VectorX<T>&   weights,  // precomputed weights for n + 1 control points on this curve
                int                 co,       // index of starting input pt in current curve
                int                 cs)       // stride of input pts in current curve
        {
            int last   = R.cols() - 1;                                          // column of range value TODO: weighing only the last column does not make much sense in the split model
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);   // one row for each input point
            VectorX<T> denom(N.rows());                                         // rational denomoninator for param of each input point

            for (int k = 0; k < N.rows(); k++)
            {
                denom(k) = (N.row(k).cwiseProduct(weights.transpose())).sum();
                Rk.row(k) = in_pts.row(co + k * cs);
            }

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                                     // ith basis functions for input pts
                         weights(i) / denom.array() *                           // rationalized
                         Rk.col(j).array()).sum();                              // input points
#else                                                                           // don't weigh domain coordinate (only range)
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
            {
                // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                for (int j = 0; j < R.cols() - 1; j++)
                    R(i, j) =
                        (N.col(i).array() *                                     // ith basis functions for input pts
                         Rk.col(j).array()).sum();                              // input points
                R(i, last) =
                    (N.col(i).array() *                                         // ith basis functions for input pts
                     weights(i) / denom.array() *                               // rationalized
                     Rk.col(last).array()).sum();                               // input points
            }
#endif

        }

        // Remove deterministic validation rows before the reusable Stage-A
        // least-squares solve.  The held-out rows remain available to the
        // Stage-B validation logger but cannot influence the initialization.
        size_t ApplyInitializerFitMask(
            SparseMatrixX<T>& Nt,
            const char* problem_label)
        {
            const char* filename =
                std::getenv("MFA_INITIALIZER_ACTIVE_INDEX_FILE");
            if (!filename || !*filename)
                filename = std::getenv("MFA_DATA_ACTIVE_INDEX_FILE");
            if (!filename || !*filename)
                return static_cast<size_t>(input.npts);

            std::ifstream stream(filename);
            if (!stream)
                throw std::runtime_error(
                    std::string("Cannot open initializer active-index file: ") +
                    filename);
            std::vector<unsigned char> active(
                static_cast<size_t>(input.npts), static_cast<unsigned char>(0));
            long long index = -1;
            size_t count = 0;
            while (stream >> index)
            {
                if (index < 0 || index >= static_cast<long long>(input.npts))
                    throw std::runtime_error(
                        "Out-of-range initializer fit-row index");
                const size_t position = static_cast<size_t>(index);
                if (!active[position])
                {
                    active[position] = 1;
                    ++count;
                }
            }
            if (!stream.eof() || count == 0)
                throw std::runtime_error(
                    "Malformed or empty initializer fit-row file");

            for (int column = 0; column < Nt.outerSize(); ++column)
                if (!active[static_cast<size_t>(column)])
                    for (typename SparseMatrixX<T>::InnerIterator it(Nt, column);
                         it; ++it)
                        it.valueRef() = T(0);
            Nt.prune(T(0));
            fmt::print(
                "{} data-only initializer: using {} of {} fit rows from {}\n",
                problem_label, count, input.npts, filename);
            return count;
        }

        // computes right hand side vector for encoding a tensor product in unified-dimensional form
        // NOTE: Does not support weights
        void RHSUnified(
            SparseMatrixX<T>&       Nt,
            MatrixX<T>&             R)
        {
            // REQUIRED for TMesh

            if (R.cols() != mfa_data.dim())
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (cols)" << endl;
            if (R.rows() != input.npts)
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (rows)" << endl;

            VectorX<T> pt_coords(mfa_data.dim());
            for (auto input_it = input.begin(); input_it != input.end(); ++input_it)
            {
                // extract coordinates in dimension min_dim<-->max_dim and place in pt_coords
                input_it.coords(pt_coords, mfa_data.min_dim, mfa_data.max_dim);

		R.row(input_it.idx()) = pt_coords;
            }


            R = Nt * R;

        }
        void RHSUnified3(
            SparseMatrixX<T>&       Nt,
            MatrixX<T>&             R)
        {
            // REQUIRED for TMesh

            if (R.cols() != mfa_data3.dim())
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (cols)" << endl;
            if (R.rows() != input.npts)
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (rows)" << endl;

            VectorX<T> pt_coords(mfa_data3.dim());
            for (auto input_it = input.begin(); input_it != input.end(); ++input_it)
            {
                // extract coordinates in dimension min_dim<-->max_dim and place in pt_coords
                input_it.coords(pt_coords, mfa_data3.min_dim, mfa_data3.max_dim);

		R.row(input_it.idx()) = pt_coords;
            }


            R = Nt * R;
        }

        void RHSUnified2(
            SparseMatrixX<T>&       Nt,
            MatrixX<T>&             R)
        {
            // REQUIRED for TMesh

            if (R.cols() != mfa_data2.dim())
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (cols)" << endl;
            if (R.rows() != input.npts)
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (rows)" << endl;

            VectorX<T> pt_coords(mfa_data2.dim());
            for (auto input_it = input.begin(); input_it != input.end(); ++input_it)
            {
                // extract coordinates in dimension min_dim<-->max_dim and place in pt_coords
                input_it.coords(pt_coords, mfa_data2.min_dim, mfa_data2.max_dim);

		R.row(input_it.idx()) = pt_coords;
            }


            R = Nt * R;
        }

        void RHSUnified_BC(
            SparseMatrixX<T>&       Nt,
            SparseMatrixX<T>&       Bt,
            MatrixX<T>&             R,
            MatrixX<T>&             R_BC)
        {
            if (R.cols() != mfa_data.dim())
            cerr << "Error: Incorrect matrix dimensions in RHSUnified (cols)" << endl;
            if (R.rows() != input.npts)
            cerr << "Error: Incorrect matrix dimensions in RHSUnified (rows)" << endl;

            VectorX<T> pt_coords(mfa_data.dim());
            VectorX<T> pt_coords_BC(mfa_data.dim());
            for (auto input_it = input.begin(); input_it != input.end(); ++input_it)
            {
                // extract coordinates in dimension min_dim<-->max_dim and place in pt_coords
                input_it.coords(pt_coords, mfa_data.min_dim, mfa_data.max_dim);
                R.row(input_it.idx()) = pt_coords;
            }
            for (auto input_it = input_BC.begin(); input_it != input_BC.end(); ++input_it)
            {
                // extract values from boundary condition data
                input_it.coords(pt_coords_BC, mfa_data.min_dim, mfa_data.max_dim);
                R_BC.row(input_it.idx()) = pt_coords_BC;
            }

            R = (Nt * R) + (Bt * R_BC);
        }

        // computes right hand side vector for encoding one tensor product of control points
        // takes subset of original input points
        // solves all dimensions at once (not seperable dimensions)
        // allows for additional constraints
        // no weights as yet
        // R is column vector of n elements, each element multiple coordinates of the input points
        void RHSTensor(
                const MatrixX<T>&       N,          // matrix of basis function coefficients
                const vector<size_t>&   start_idxs, // starting indices of subset of domain points
                const vector<size_t>&   end_idxs,   // ending indices of subset of domain points
                const TensorProduct<T>& t,          // current tensor product
                MatrixX<T>&             R)          // (output) residual matrix allocated by caller
        {
            VectorXi ndom_pts(dom_dim);
            VectorXi dom_starts(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];
            }

            // fill Rk, the matrix of input points
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);           // one row for each input point
            VolIterator vol_iter(ndom_pts, dom_starts, input.ndom_pts());                 // iterator over input points
            VectorXi ijk(dom_dim);
            while (!vol_iter.done())
            {
                vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);
                Rk.row(vol_iter.cur_iter()) = input.domain.block(vol_iter.ijk_idx(ijk), mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
                vol_iter.incr_iter();
            }

            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                 // ith basis functions for input pts
                         Rk.col(j).array()).sum();          // input points

        }

        // Checks quantities needed for approximation
        void Quants(
                const VectorXi& nctrl_pts,      // number of control points
                VectorXi&       n,              // (output) number of control point spans in each dim
                VectorXi&       m)              // (output) number of input data point spans in each dim
        {
            if (mfa_data.p.size() != input.ndom_pts().size())
            {
                fprintf(stderr, "Error: Encode() size of p must equal size of ndom_pts\n");
                exit(1);
            }
            for (size_t i = 0; i < mfa_data.p.size(); i++)
            {
                if (nctrl_pts(i) <= mfa_data.p(i))
                {
                    fprintf(stderr, "Error: Encode() number of control points in dimension %ld "
                            "must be at least p + 1 for dimension %ld\n", i, i);
                    exit(1);
                }
                if (nctrl_pts(i) > input.ndom_pts(i))
                {
                    fprintf(stderr, "Warning: Encode() number of control points (%d) in dimension %ld "
                            "exceeds number of input data points (%d) in dimension %ld.\n", nctrl_pts(i), i, input.ndom_pts(i), i);
                }
            }

            n.resize(mfa_data.p.size());
            m.resize(mfa_data.p.size());
            for (size_t i = 0; i < mfa_data.p.size(); i++)
            {
                n(i)        =  nctrl_pts(i) - 1;
                m(i)        =  input.ndom_pts(i)  - 1;
            }
        }

        // solves for one curve of control points
        // outputs go to specified control points and weights matrix and vector rather than default mfa
        void CtrlCurve(
                const MatrixX<T>&   N,                  // basis functions for current dimension
                const MatrixX<T>&   NtN,                // Nt * N
                MatrixX<T>&         R,                  // (output) residual matrix for current dimension and curve
                MatrixX<T>&         P,                  // (output) solved points for current dimension and curve
                size_t              k,                  // current dimension
                size_t              co,                 // starting ofst for reading domain pts
                size_t              cs,                 // stride for reading domain points
                size_t              to,                 // starting ofst for writing control pts
                MatrixX<T>&         temp_ctrl0,         // first temporary control points buffer
                MatrixX<T>&         temp_ctrl1,         // second temporary control points buffer
                int                 curve_id,           // debugging
                MatrixX<T>&         ctrl_pts,           // (output) control points
                VectorX<T>&         weights,            // (output) weights
                bool                weighted = true)    // solve for and use weights
        {
            // solve for weights
            // TODO: avoid copying into Q by passing temp_ctrl0, temp_ctrl1, co, cs to Weights()
            // TODO: check that this is right, using co and cs for copying control points and domain points
            MatrixX<T> Q;
            Q.resize(input.ndom_pts(k), ctrl_pts.cols());
            if (k == 0)
            {
                for (auto i = 0; i < input.ndom_pts(k); i++)
                    Q.row(i) = input.domain.block(co + i * cs, mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
            }
            else if (k % 2)
            {
                for (auto i = 0; i < input.ndom_pts(k); i++)
                    Q.row(i) = temp_ctrl0.row(co + i * cs);
            }
            else
            {
                for (auto i = 0; i < input.ndom_pts(k); i++)
                    Q.row(i) = temp_ctrl1.row(co + i * cs);
            }

            VectorX<T> temp_weights = VectorX<T>::Ones(N.cols());

#ifndef MFA_NO_WEIGHTS

            if (weighted)
                if (k == dom_dim - 1)                               // only during last dimension of separable iteration over dimensions
                    Weights(k, Q, N, NtN, curve_id, temp_weights);      // solve for weights

#endif

            // compute R
            // first dimension reads from domain
            // subsequent dims alternate reading temp_ctrl0 and temp_ctrl1
            // even dim reads temp_ctrl1, odd dim reads temp_ctrl0; opposite of writing order
            // because what was written in the previous dimension is read in the current one
            if (k == 0)
                RHS(k, N, R, temp_weights, co);                         // input points = default domain
            else if (k % 2)
                RHS(k, temp_ctrl0, N, R, temp_weights, co, cs);         // input points = temp_ctrl0
            else
                RHS(k, temp_ctrl1, N, R, temp_weights, co, cs);         // input points = temp_ctrl1

            // rationalize NtN, ie, weigh the basis function coefficients
            MatrixX<T> NtN_rat = NtN;
            mfa_data.Rationalize(k, temp_weights, N, NtN_rat);

            // solve for P
#ifdef WEIGH_ALL_DIMS                                   // weigh all dimensions
            P = NtN_rat.ldlt().solve(R);
#else                                                   // don't weigh domain coordinate (only range)
            // TODO: avoid 2 solves?
            MatrixX<T> P2(P.rows(), P.cols());
            P = NtN.ldlt().solve(R);                            // nonrational domain coordinates
            P2 = NtN_rat.ldlt().solve(R);                       // rational range coordinate
            for (auto i = 0; i < P.rows(); i++)
                P(i, P.cols() - 1) = P2(i, P.cols() - 1);
#endif

            // append points from P to control points that will become inputs for next dimension
            // TODO: any way to avoid this?
            CopyCtrl(P, k, co, cs, to, ctrl_pts, temp_ctrl0, temp_ctrl1);

            // copy weights of final dimension to mfa
            if (k == dom_dim - 1)
            {
                for (auto i = 0; i < temp_weights.size(); i++)
                    weights(to + i * cs) = temp_weights(i);
            }
        }

        // append solved control points from P to become inputs for next dimension
        // TODO: any way to avoid this copy?
        // last dimension gets copied to final control points
        // This version specifies a location for ctrl_pts rather than default one in mfa
        // previous dimensions get copied to alternating double buffers
        void CopyCtrl(
                const MatrixX<T>&   P,              // solved points for current dimension and curve
                int                 k,              // current dimension
                size_t              co,             // starting offset for reading domain points
                size_t              cs,             // stride for reading domain points
                size_t              to,             // starting offset for writing control points
                MatrixX<T>&         ctrl_pts,       // (output) control points
                MatrixX<T>&         temp_ctrl0,     // (output) first temporary control points buffer
                MatrixX<T>&         temp_ctrl1)     // (output) second temporary control points buffer
        {
            int ndims = dom_dim;

            // if there is only one dim, copy straight to output
            if (ndims == 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    ctrl_pts.row(to + i * cs) = P.row(i);
            }
            // first dim copied from domain to temp_ctrl0
            else if (k == 0)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl0.row(to + i * cs) = P.row(i);
            }
            // even numbered dims (but not the last one) copied from P to temp_ctrl0
            else if (k % 2 == 0 && k < ndims - 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl0.row(to + i * cs) = P.row(i);
            }
            // odd numbered dims (but not the last one) copied from P to temp_ctrl1
            else if (k % 2 == 1 && k < ndims - 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl1.row(to + i * cs) = P.row(i);
            }
            // final dim if even is copied from temp_ctrl1 to ctrl_pts
            else if (k == ndims - 1 && k % 2 == 0)
            {
                for (int i = 0; i < P.rows(); i++)
                    ctrl_pts.row(to + i * cs) = P.row(i);
            }
            // final dim if odd is copied from temp_ctrl0 to ctrl_pts
            else if (k == ndims - 1 && k % 2 == 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    ctrl_pts.row(to + i * cs) = P.row(i);
            }
        }

        // append solved control points from P to become inputs for next dimension
        // TODO: any way to avoid this copy?
        // last dimension gets copied to final control points
        // previous dimensions get copied to alternating double buffers
        void CopyCtrl(
                const MatrixX<T>&   P,              // solved points for current dimension and curve
                int                 k,              // current dimension
                size_t              co,             // starting offset for reading domain points
                size_t              cs,             // stride for reading domain points
                size_t              to,             // starting offset for writing control points
                TensorProduct<T>&   tensor,         // (output) tensor product containing result
                MatrixX<T>&         temp_ctrl0,     // first temporary control points buffer
                MatrixX<T>&         temp_ctrl1)     // second temporary control points buffer
        {
            int ndims = dom_dim;

            // if there is only one dim, copy straight to output
            if (ndims == 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    tensor.ctrl_pts.row(to + i * cs) = P.row(i);
            }
            // first dim copied from domain to temp_ctrl0
            else if (k == 0)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl0.row(to + i * cs) = P.row(i);
            }
            // even numbered dims (but not the last one) copied from P to temp_ctrl0
            else if (k % 2 == 0 && k < ndims - 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl0.row(to + i * cs) = P.row(i);
            }
            // odd numbered dims (but not the last one) copied from P to temp_ctrl1
            else if (k % 2 == 1 && k < ndims - 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl1.row(to + i * cs) = P.row(i);
            }
            // final dim if even is copied from temp_ctrl1 to ctrl_pts
            else if (k == ndims - 1 && k % 2 == 0)
            {
                for (int i = 0; i < P.rows(); i++)
                    tensor.ctrl_pts.row(to + i * cs) = P.row(i);
            }
            // final dim if odd is copied from temp_ctrl0 to ctrl_pts
            else if (k == ndims - 1 && k % 2 == 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    tensor.ctrl_pts.row(to + i * cs) = P.row(i);
            }
        }

        // computes new knots to be inserted into a curve
        // for each current knot span where the error is greater than the limit, finds the domain point
        // where the error is greatest and adds the knot at that parameter value
        //
        // this version takes a set of control points as input instead of mfa_data.ctrl_pts
        int ErrorCurve(
                size_t                  k,          // current dimension
                const TensorProduct<T>& tensor,     // current tensor product
                size_t                  co,         // starting ofst for reading domain pts
                const MatrixX<T>&       ctrl_pts,   // control points
                const VectorX<T>&       weights,    // weights associated with control points
                VectorX<T>              extents,    // extents in each dimension, for normalizing error (size 0 means do not normalize)
                set<int>&               err_spans,  // (output) spans with error greater than err_limit
                T                       err_limit)  // max allowable error
        {
            mfa::Decoder<T> decoder(mfa_data, verbose);
            int pt_dim = tensor.ctrl_pts.cols();            // control point dimensonality
            VectorX<T> cpt(pt_dim);                         // decoded curve point
            int nerr = 0;                                   // number of points with error greater than err_limit
            int span = mfa_data.p[k];                            // current knot span of the domain point being checked
            if (!extents.size())
                extents = VectorX<T>::Ones(input.domain.cols());

            for (auto i = 0; i < input.ndom_pts(k); i++)      // all domain points in the curve
            {
                while (mfa_data.tmesh.all_knots[k][span + 1] < 1.0 && mfa_data.tmesh.all_knots[k][span + 1] <= input.params->param_grid[k][i])
                    span++;

                decoder.CurvePt(k, input.params->param_grid[k][i], ctrl_pts, weights, tensor, cpt);


                // error
                T max_err = 0.0;
                for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
                {
                    T err = fabs(cpt(j) - input.domain(co + i * input.g.ds[k], mfa_data.min_dim + j)) / extents(mfa_data.min_dim + j);
                    max_err = err > max_err ? err : max_err;
                }

                if (max_err > err_limit &&
                        // don't allow more control points than input points
                        mfa_data.tmesh.tensor_prods.size() == 1 &&
                        tensor.nctrl_pts(k) + err_spans.size() <= input.ndom_pts(k))
                {

                    // don't duplicate spans
                    set<int>::iterator it = err_spans.find(span);
                    if (!err_spans.size() || it == err_spans.end())
                    {
                        // ensure there would be a domain point in both halves of the span if it were split
                        bool split_left = false;
                        for (auto j = i; input.params->param_grid[k][j] >= mfa_data.tmesh.all_knots[k][span]; j--)
                            if (input.params->param_grid[k][j] < (mfa_data.tmesh.all_knots[k][span] + mfa_data.tmesh.all_knots[k][span + 1]) / 2.0)
                            {
                                split_left = true;
                                break;
                            }
                        bool split_right = false;
                        for (auto j = i; input.params->param_grid[k][j] < mfa_data.tmesh.all_knots[k][span + 1]; j++)
                            if (input.params->param_grid[k][j] >= (mfa_data.tmesh.all_knots[k][span] + mfa_data.tmesh.all_knots[k][span + 1]) / 2.0)
                            {
                                split_right = true;
                                break;
                            }
                        // mark the span and count the point if the span can (later) be split
                        if (split_left && split_right)
                            err_spans.insert(it, span);
                    }
                    // count the point in the total even if the span is not marked for splitting
                    // total used to find worst curve, defined as the curve with the most domain points in
                    // error (including multiple domain points per span and points in spans that can't be
                    // split further)
                    nerr++;
                }
            }

            return nerr;
        }

#ifdef MFA_TMESH

        // refines a T-mesh at a given parent level
        // returns true no change in knots; all tensors at the parent level are done
        bool Refine(
                int                         parent_level,                       // level of parent tensors to refine
                int                         child_level,                        // level of children to create
                T                           err_limit,                          // max allowable error
                const VectorX<T>&           extents,                            // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                         pad,                                // padding (per side) for all tensors
                int                         edge_pad,                           // extra padding (per side) for tensor at the global edge
                vector<TensorProduct<T>>&   new_tensors)                        // (output) new tensors scheduled to be added
        {
            // typing shortcuts
            Tmesh<T>&                   tmesh                   = mfa_data.tmesh;
            vector<vector<T>>&          all_knots               = tmesh.all_knots;
            vector<vector<int>>&        all_knot_levels         = tmesh.all_knot_levels;
            vector<vector<ParamIdx>>&   all_knot_param_idxs     = tmesh.all_knot_param_idxs;
            vector<TensorProduct<T>>&   tensor_prods            = tmesh.tensor_prods;
            VectorXi&                   p                       = mfa_data.p;

            vector<vector<KnotIdx>>     inserted_knot_idxs(dom_dim);            // indices in each dim. of inserted knots in full knot vector after insertion
            vector<vector<T>>           inserted_knots(dom_dim);                // knots to be inserted in each dim.
            vector<TensorIdx>           parent_tensor_idxs;                     // tensors having knots inserted

            VectorX<T> myextents = extents.size() ? extents : VectorX<T>::Ones(tensor_prods[0].ctrl_pts.cols());
            ErrorStats<T> error_stats;

            // find new knots
            mfa::NewKnots<T> nk(mfa_data, input);

            // vectors of new_nctrl_pts, new_ctrl_pts, new_weights, one instance for each knot to be inserted
            vector<VectorXi>    new_nctrl_pts;
            vector<MatrixX<T>>  new_ctrl_pts;
            vector<VectorX<T>>  new_weights;

            // timing
            double error_spans_time = MPI_Wtime();

            // check all knots spans for error
            bool done = nk.AllErrorSpans(
                    parent_level,
                    myextents,
                    err_limit,
                    false,
                    new_tensors,
                    parent_tensor_idxs,
                    inserted_knot_idxs,
                    inserted_knots,
                    error_stats);

            if (done)                                                           // nothing inserted
                return true;

            int n_insertions = parent_tensor_idxs.size();                       // number of knots to insert
            for (auto j = 0; j < dom_dim; j++)
                assert(inserted_knot_idxs[j].size() == n_insertions &&
                        inserted_knots[j].size() == n_insertions);

            vector<bool> inserted(dom_dim);                                     // whether the current insertion succeeded (in each dim)

            // timing
            error_spans_time    = MPI_Wtime() - error_spans_time;
            double insert_time  = MPI_Wtime();


            for (auto i = 0; i < n_insertions; i++)                             // for all knots to be inserted
            {
                // Require the parent tensor to belong to the current refinement level.
                if (tensor_prods[parent_tensor_idxs[i]].level > parent_level)
                    throw MFAError(fmt::format("Error: Refine(): insertion index {} with parent tensor idx {} "
                            "at level {}, but parent_level is {}. This should not happen.\n",
                            i, parent_tensor_idxs[i], tensor_prods[parent_tensor_idxs[i]].level, parent_level));


                // insert the new knot into tmesh all_knots
                // NB: insert_knot adjusts knot_mins, maxs of existing tensors, meaning
                // no further adjustment to existing tensors should be done here
                int retval;
                bool new_candidate = false;                                     // make new candidate tensor for this knot
                for (auto j = 0; j < dom_dim; j++)
                {
                    inserted[j] = false;
                    retval = tmesh.insert_knot(
                                j,
                                child_level,
                                inserted_knots[j][i],
                                input.params->param_grid,
                                inserted_knot_idxs[j][i]);

                    if (retval == 1 || retval == 2)
                    {
                        inserted[j] = true;
                        new_candidate = true;
                    }
                }

                // it's possible that all components of the knot exist separately but not together
                // in this case check if the full-dimension knot is already part of a candidate tensor before skipping
                if (find(inserted.begin(), inserted.end(), true) == inserted.end())     // knot exists in all dimensions separately
                {
                    vector<KnotIdx> knot(dom_dim);                                      // full-dim. knot
                    for (auto j = 0; j < dom_dim; j++)
                        knot[j] = inserted_knot_idxs[j][i];

                    TensorIdx k;
                    for (k = 0; k < new_tensors.size(); k++)                            // check all candidate tensors
                    {
                        auto& c = new_tensors[k];
                        if (tmesh.in(knot, c.knot_mins, c.knot_maxs))
                            break;
                    }
                    if (k == new_tensors.size())                                        // no candidates contain the knot
                            new_candidate = true;
                }

                if (!new_candidate)
                    continue;

                // make a candidate tensor with the new knot plus the pad
                TensorProduct<T> c(dom_dim);
                c.level                 = child_level;
                c.parent                = parent_tensor_idxs[i];
                c.parent_exists         = true;
                TensorProduct<T>& pt    = tensor_prods[parent_tensor_idxs[i]];             // parent tensor of the candidate tensor
                std::vector<KnotIdx> inserted_knot_idx(dom_dim);
                for (auto j = 0; j < dom_dim; j++)
                    inserted_knot_idx[j] = inserted_knot_idxs[j][i];
                tmesh.make_candidate(inserted_knot_idx, c, pt, pad, edge_pad);

                // adjust knot mins, maxs of tensors to be added so far because of inserted knots
                for (auto tidx = 0; tidx < new_tensors.size(); tidx++)          // for all tensors scheduled to be added so far
                {
                    auto& t = new_tensors[tidx];

                    // adjust previously scheduled tensor knot mins, maxs for new knot insertion
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        if (inserted[j] && inserted_knot_idxs[j][i] <= t.knot_mins[j])
                            t.knot_mins[j]++;
                        if (inserted[j] && inserted_knot_idxs[j][i] <= t.knot_maxs[j])
                            t.knot_maxs[j]++;
                    }
                }

                // Keep the candidate's knot bounds within its parent level.
                if (!tmesh.check_knot_edge_level(c, tensor_prods[c.parent].level))
                    throw MFAError(fmt::format("Refine() 1: knot mins/maxs exceed parent level\n"));

                // check/adjust candidate knot mins and maxs subset and intersection against tensors to be added so far
                while (1)
                {
                    bool changed_new_tensors = false;                               // new_tensors changed and will need to be (re)checked
                    bool add    = true;                                             // the candidate tensor c should be added to the new tensors

                    for (auto tidx = 0; tidx < new_tensors.size(); tidx++)          // for all tensors scheduled to be added so far
                    {
                        auto& t = new_tensors[tidx];
                        if (t.level < 0)                                            // t was marked as invalid and should be skipped
                            continue;

                        // candidate c is a subset of an already scheduled tensor t
                        // keep t, invalidate c
                        if (tmesh.subset(c.knot_mins, c.knot_maxs, t.knot_mins, t.knot_maxs))
                        {
                            t.level             = child_level;
                            add                 = false;
                            break;
                        }

                        // an already scheduled tensor is a subset of the candidate
                        // keep c, invalidate t
                        else if (tmesh.subset(t.knot_mins, t.knot_maxs, c.knot_mins, c.knot_maxs))
                        {
                            if (t.parent != c.parent)
                            {
                                // TODO: not sure if this should be an error, or if the parent should be reset
                                fmt::print(stderr, "Error: already scheduled tensor is a subset of the candidate tensor, but they have different parents. This should not happen\n");
                                abort();
                            }
                            t.level             = -1;
                            add                 = true;
                            changed_new_tensors = true;
                        }

                        // candidate intersects an already scheduled tensor, to within some proximity
                        // merge c into t, update and keep t, invalidate c
                        // pad - 1 allows remainders of size pad, which is what we want
                        else if (tmesh.intersect(c, t, pad - 1))
                        {
                            if (c.parent == t.parent)                           // only merge tensors refined from the same parent
                            {
                                tmesh.merge_tensors(t, c, pad, edge_pad);       // merge c into t, updating the knot mins, maxs of t
                                t.level             = child_level;
                                add                 = false;
                                break;
                            }
                        }

                    }       // for all tensors scheduled to be added so far

                    // schedule the tensor to be added
                    if (add)
                    {
                        new_tensors.push_back(c);

                        // Keep the scheduled candidate's knot bounds within its parent level.
                        if (!tmesh.check_knot_edge_level(c, tensor_prods[c.parent].level))
                            throw MFAError(fmt::format("Refine() 2: knot mins/maxs exceed parent level\n"));
                    }

                    if (!changed_new_tensors)
                        break;
                }   // while !all_done
            }   // for all knots to be inserted

            // timing
            fmt::print(stderr, "error spans time:       {:.3e} s.\n", error_spans_time);

            return false;
        }

        // final double-check that new tensor products do not intersect each other within some pad amount
        // check all tensors to be added against each other
        void CheckNewTensors(
                vector<TensorProduct<T>>&   new_tensors,                        // new tensors scheduled to be added
                int                         pad,                                // padding for all tensors
                int                         edge_pad)                           // extra padding for tensor at the global edge
        {
            Tmesh<T>& tmesh = mfa_data.tmesh;
            VectorXi& p     = mfa_data.p;

            while (1)
            {
                bool changed_new_tensors = false;                               // new_tensors changed and will need to be (re)checked

                for (auto cidx = 0; cidx < new_tensors.size(); cidx++)          // for all candidate tensors from tensors scheduled to be added so far
                {
                    auto& c = new_tensors[cidx];
                    if (c.level < 0)                                            // c was marked as invalid and should be skipped
                        continue;

                    for (auto tidx = 0; tidx < new_tensors.size(); tidx++)      // for all tensors scheduled to be added so far
                    {
                        auto& t = new_tensors[tidx];
                        if (cidx == tidx || t.level < 0)                        // t was marked as invalid and should be skipped
                            continue;

                        // candidate c is a subset of an already scheduled tensor t
                        // keep t, invalidate c
                        if (tmesh.subset(c.knot_mins, c.knot_maxs, t.knot_mins, t.knot_maxs))
                        {
                            c.level             = -1;
                            break;
                        }

                        // an already scheduled tensor is a subset of the candidate
                        // keep c, invalidate t
                        else if (tmesh.subset(t.knot_mins, t.knot_maxs, c.knot_mins, c.knot_maxs))
                        {
                            if (t.parent != c.parent)
                            {
                                // TODO: not sure if this should be an error, or if the parent should be reset
                                fmt::print(stderr, "Error: already scheduled tensor is a subset of the candidate tensor, but they have different parents. This should not happen\n");
                                abort();
                            }
                            t.level             = -1;
                        }

                        // candidate intersects an already scheduled tensor, to within some proximity
                        // merge c into to t, update and keep t, invalidate c
                        // pad - 1 allows remainders of size pad, which is what we want
                        else if (tmesh.intersect(c, t, pad - 1))
                        {
                            if (c.parent == t.parent)                           // only merge tensors refined from the same parent
                            {
                                tmesh.merge_tensors(t, c, pad, edge_pad);       // merge c into t, updating the knot mins, maxs of t
                                c.level                 = -1;
                                changed_new_tensors     = true;
                                break;
                            }
                        }

                    }   // for all tensors tidx
                }   // for all tensors cidx

                if (!changed_new_tensors)
                    break;
            }   // while !all_done
        }

        // appends and encodes a vector of new tensor products
        // returns whether all tensors were successfully added
        bool AddNewTensors(
                vector<TensorProduct<T>>& new_tensors,
                int pad)                                        // required distance between new tensors and existing tensors
        {
            // typing shortcuts
            auto&   tmesh                   = mfa_data.tmesh;
            auto&   tensor_prods            = tmesh.tensor_prods;
            auto&   p                       = mfa_data.p;

            bool retval = true;

            for (auto k = 0; k < new_tensors.size(); k++)       // for all tensors to append
            {
                auto& t = new_tensors[k];

                if (t.level < 0)                                // tensor was removed from the schedule
                    continue;

                // check one last time intersection of new tensor t and other existing tensors e in new_tensors would result in too small of a tensor
                for (auto j = 0; j < tensor_prods.size(); j++)
                {
                    auto& e = tensor_prods[j];                  // existing tensor

                    // outright intersection that is too near to edge of existing tensor
                    // can't handle this case at this point in the algorithm
                    // need to invalidate the candidate, mark intersected existing tensor as not done
                    // check its level, hopefully no higher up than the parent level
                    // and don't increase the parent level
                    // wait to pick this candidate up in the next iteration

                    // check for intersection between t (tensor to be added) and e (existing tensor)
                    std::vector<KnotIdx> inter_mins(t.knot_mins.size());
                    std::vector<KnotIdx> inter_maxs(t.knot_maxs.size());
                    if (tmesh.intersects(t.knot_mins, t.knot_maxs, e.knot_mins, e.knot_maxs, inter_mins, inter_maxs))
                    {
                        // check intersection amount in each dimension
                        // since the candidate is the deepest level, consider all knot indices using a simple subtraction (not Tmesh::knot_idx_dist()
                        bool found_inter = false;
                        for (auto i = 0; i < dom_dim; i++)
                        {
                            // check t.knot_mins against both sides of e
                            if (t.level > 0 && t.knot_mins[i] > e.knot_mins[i] && t.knot_mins[i] < e.knot_maxs[i])
                            {
                                if (t.knot_mins[i] - e.knot_mins[i] < pad || e.knot_maxs[i] - t.knot_mins[i] < pad)
                                {
                                    found_inter = true;
                                    break;
                                }
                            }
                            // check t.knot_maxs against both sides of e
                            if (t.level > 0 && !found_inter && t.knot_maxs[i] > e.knot_mins[i] && t.knot_maxs[i] < e.knot_maxs[i])
                            {
                                if (t.knot_maxs[i] - e.knot_mins[i] < pad || e.knot_maxs[i] - t.knot_maxs[i] < pad)
                                {
                                    found_inter = true;
                                    break;
                                }
                            }
                        }

                        if (found_inter)
                        {

                            // invalidate this tensor
                            t.level = -1;
                            retval  = false;
                        }

                        if (t.level < 0)
                            e.done = false;                 // any existing tensor intersecting a tensor that could not be added should be rechecked next time
                    }       // e intersects t

                    // check for nearness to existing tensor without actually intersecting
                    // in this case, merge e and t, enlarging to cover e
                    // pad - 1 allows remainders of size pad to remain unmerged, which is what we want
                    if (!tmesh.intersect(t, e, 0, true, false) && tmesh.intersect(t, e, pad - 1, true, false))
                    {
                        // merge e (existing tensor) into t (tensor to be added)
                        // ie, enlarge t to cover e
                        tmesh.merge_tensors(t, e, -1, 0);

                    }
                }

                if (t.level < 0)
                    continue;


                // append the tensor
                int tensor_idx = tmesh.append_tensor(t.knot_mins, t.knot_maxs, t.level);


                // Verify input-point span coverage before solving.
                mfa::NewKnots<T> nk(mfa_data, input);
                if (!nk.CheckAllSpans())
                    throw MFAError(fmt::format("AddNewTensors(): Error: failed checking all spans for input points\n"));

                // Verify knot and control-point counts for every tensor.
                for (auto j = 0; j < tensor_prods.size(); j++)
                    if (!tmesh.check_num_knots_ctrl_pts(j))
                        throw MFAError(fmt::format("AddNewTensors(): number of knots and control points do not agree\n"));

                // Enforce the minimum control-point count for every tensor.
                for (auto i = 0; i < tensor_prods.size(); i++)
                {
                    if (!tmesh.check_num_ctrl_degree(i, 0))
                    {
                        auto& t = tensor_prods[i];
                        fmt::print(stderr, "Error: AddNewTensors(): After appending, tensor tidx {} knot_mins [{}] knot_maxs [{}] level {} has fewer than p control points.\n",
                                i, fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level);
                        fmt::print(stderr, "\nAddNewTensors(): T-mesh after appending tensor:\n");
                        tmesh.print(false, false, false, false);
                        abort();
                    }
                }

                // solve for new control points

#ifdef MFA_ENCODE_LOCAL_SEPARABLE

                EncodeTensorLocalSeparable(tensor_idx);

#else

                EncodeTensorLocalUnified(tensor_idx);

#endif
            }   // for all tensors k to append

            return retval;
        }

        // updates and encodes existing tensors if they match any in a vector of new tensor products
        void UpdateExistingTensors(vector<TensorProduct<T>>& new_tensors)
        {
            // typing shortcuts
            auto&   tmesh                   = mfa_data.tmesh;
            auto&   tensor_prods            = tmesh.tensor_prods;
            auto&   dom_dim                 = mfa_data.dom_dim;
            auto&   p                       = mfa_data.p;

            // Reuse the span validator while updating tensors.
            mfa::NewKnots<T> nk(mfa_data, input);

            for (auto k = 0; k < new_tensors.size(); k++)
            {
                auto& t = new_tensors[k];

                if (t.level < 0)                        // tensor was removed from the schedule
                    continue;

                int tensor_idx = tmesh.update_tensor(t.knot_mins, t.knot_maxs, t.level);

                if (tensor_idx < 0)
                    continue;


                // Verify input-point span coverage before solving.
                if (!nk.CheckAllSpans())
                    throw MFAError(fmt::format("UpdateExistingTensors(): Error: failed checking all spans for input points\n"));

                // Verify knot and control-point counts for every tensor.
                for (auto j = 0; j < tensor_prods.size(); j++)
                    if (!tmesh.check_num_knots_ctrl_pts(j))
                        throw MFAError(fmt::format("UpdateExistingTensors(): number of knots and control points do not agree\n"));

                // Enforce the minimum control-point count for every tensor.
                for (auto i = 0; i < tensor_prods.size(); i++)
                {
                    if (!tmesh.check_num_ctrl_degree(i, 0))
                    {
                        auto& t = tensor_prods[k];
                        fmt::print(stderr, "Error: UpdateExistingTensors(): After appending tensor k {} one of the tensors has fewer than p control points. This should not happen\n", k);
                        fmt::print(stderr, "New tensor knot_mins [{}] knot_maxs[{}] level {} parent tensor {}\n",
                                fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level, t.parent);
                        fmt::print(stderr, "\nUpdateExistingTensors(): T-mesh after appending tensor k {}\n", k);
                        tmesh.print(true, true, false, false);
                        abort();
                    }
                }

                // solve for new control points
#ifdef MFA_ENCODE_LOCAL_SEPARABLE
                EncodeTensorLocalSeparable(tensor_idx);
#else
                EncodeTensorLocalUnified(tensor_idx);
#endif
                t.level = -1;                       // remove this tensor from the new tensors
            }
        }

        // constraint control points and corresponding anchors for local solve
        // helper function for EncodeTensorLocalSeparable and EncodeTensorLocalUnified
        // this version checks all tensors, slower than looking at prev/next, but reliably finds all constraints
        void LocalSolveAllConstraints(
                const TensorProduct<T>&     tc,                 // current tensor product being solved
                MatrixX<T>&                 ctrl_pts,           // (output) constraining control points
                vector<vector<KnotIdx>>&    anchors,            // (output) corresponding anchors
                vector<TensorIdx>&          t_idx_anchors)      // (output) tensors containing corresponding anchors
        {
            const Tmesh<T>&         tmesh   = mfa_data.tmesh;
            int                     cols    = tc.ctrl_pts.cols();

            // mins, maxs of tc padded by degree p
            vector<KnotIdx> tc_pad_mins(dom_dim);
            vector<KnotIdx> tc_pad_maxs(dom_dim);

            // intersection of tc padded by degree p with tensor being visited
            vector<KnotIdx> intersect_mins(dom_dim);
            vector<KnotIdx> intersect_maxs(dom_dim);

            // get required sizes

            int rows = 0;                                       // number of rows required in ctrl_pts
            VectorXi npts = VectorXi::Zero(dom_dim);

            for (auto k = 0; k < tmesh.tensor_prods.size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[k];
                if (&t == &tc)
                    continue;
                if (t.level > tc.level)
                    continue;

                // pad mins and maxs of tc
                for (auto i = 0; i < dom_dim; i++)
                {
                    int p = mfa_data.p(i);
                    tmesh.knot_idx_ofst(t, tc.knot_mins[i], -p, i, true, tc_pad_mins[i]);
                    tmesh.knot_idx_ofst(t, tc.knot_maxs[i], p, i, true, tc_pad_maxs[i]);
                }

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < dom_dim; i++)
                    {
                        // compute npts
                        if (mfa_data.p(i) % 2)              // odd degree
                            npts(i) = tmesh.knot_idx_dist(t, intersect_mins[i], intersect_maxs[i], i, true);
                        else                                // even degree
                            npts(i) = tmesh.knot_idx_dist(t, intersect_mins[i], intersect_maxs[i], i, false);
                    }
                }
                rows += npts.prod();
            }       // for all tensor products

            ctrl_pts.resize(rows, cols);

            anchors.resize(rows);
            t_idx_anchors.resize(rows);

            // get control points and anchors
            int cur_row = 0;
            VectorXi sub_starts(dom_dim);
            VectorXi sub_npts(dom_dim);
            VectorXi all_npts(dom_dim);
            vector<KnotIdx> anchor(dom_dim);           // one anchor
            for (auto k = 0; k < tmesh.tensor_prods.size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[k];
                if (&t == &tc)
                    continue;
                if (t.level > tc.level)
                    continue;

                // pad mins and maxs of tc
                for (auto i = 0; i < dom_dim; i++)
                {
                    int p = mfa_data.p(i);
                    tmesh.knot_idx_ofst(t, tc.knot_mins[i], -p, i, true, tc_pad_mins[i]);
                    tmesh.knot_idx_ofst(t, tc.knot_maxs[i], p, i, true, tc_pad_maxs[i]);
                }

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        // compute sub_starts, sub_npts, all_npts
                        sub_starts(i) = tmesh.knot_idx_dist(t, t.knot_mins[i], intersect_mins[i], i, false);
                        if (t.knot_mins[i] == 0)
                        sub_starts(i) -= (p + 1) / 2;
                        if (mfa_data.p(i) % 2)              // odd degree
                            sub_npts(i) = tmesh.knot_idx_dist(t, intersect_mins[i], intersect_maxs[i], i, true);
                        else                                // even degree
                            sub_npts(i) = tmesh.knot_idx_dist(t, intersect_mins[i], intersect_maxs[i], i, false);
                        all_npts(i) = t.nctrl_pts(i);
                    }

                    VolIterator voliter(sub_npts, sub_starts, all_npts);
                    VectorXi ijk(dom_dim);
                    while (!voliter.done())
                    {
                        // skip MFA_NAW control points (used in odd degree cases)
                        if (t.weights(voliter.sub_full_idx(voliter.cur_iter())) != MFA_NAW)
                        {
                            // control point
                            ctrl_pts.row(cur_row) = t.ctrl_pts.row(voliter.sub_full_idx(voliter.cur_iter()));

                            // anchor
                            anchors[cur_row].resize(dom_dim);
                            voliter.idx_ijk(voliter.cur_iter(), ijk);
                            mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);
                            for (auto i = 0; i < dom_dim; i++)
                                anchors[cur_row][i] = anchor[i];
                            t_idx_anchors[cur_row] = k;
                            cur_row++;
                        }
                        voliter.incr_iter();
                    }   // voliter
                }   // intersects
            }   // for all tensor products

            if (cur_row < ctrl_pts.rows())          // not all control points were used because of skipping MFA_NAW
            {
                ctrl_pts.conservativeResize(cur_row, cols);
                anchors.resize(cur_row);
                t_idx_anchors.resize(cur_row);
            }
        }

#endif      // MFA_TMESH

        // 1d encoding and 1d decoding
        // adds knots error spans from all curves in all directions (into a set)
        // adds knots in middles of spans that have error higher than the limit
        // returns true if done, ie, no knots are inserted
        bool OrigNewKnots_curve(
                vector<vector<T>>&  new_knots,                              // (output) new knots
                T                   err_limit,                              // max allowable error
                const VectorX<T>&   extents,                                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 iter)                                   // iteration number of caller (for debugging)
        {
            int     pt_dim          = mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols();    // control point dimensonality
            size_t  tot_nnew_knots  = 0;                                            // total number of new knots found
            new_knots.resize(dom_dim);
            vector<vector<int>> new_levels(dom_dim);

            for (auto& t : mfa_data.tmesh.tensor_prods)                             // for all tensor products in the tmesh
            {
                // check and assign main quantities
                VectorXi n = t.nctrl_pts - VectorXi::Ones(dom_dim);        // number of control point spans in each domain dim
                VectorXi m = input.ndom_pts()  - VectorXi::Ones(dom_dim);    // number of input data point spans in each domain dim

                // resize control points and weights
                t.ctrl_pts.resize(t.nctrl_pts.prod(), mfa_data.max_dim - mfa_data.min_dim + 1);
                t.weights = VectorX<T>::Ones(t.ctrl_pts.rows());

                for (size_t k = 0; k < dom_dim; k++)                       // for all domain dimensions
                {
                    new_knots[k].resize(0);
                    new_levels[k].resize(0);

                    // for now set weights to 1, TODO: get weights from elsewhere
                    // NB: weights are for all n + 1 control points, not just the n -1 interior ones
                    VectorX<T> weights = VectorX<T>::Ones(n(k) + 1);

                    // temporary control points for one curve
                    MatrixX<T> temp_ctrl = MatrixX<T>::Zero(t.nctrl_pts(k), pt_dim);

                    // error spans for one curve and for worst curve
                    set<int> err_spans;

                    // maximum number of domain points with error greater than err_limit and their curves
                    size_t max_nerr     =  0;

                    // N is a matrix of (m + 1) x (n + 1) scalars that are the basis function coefficients
                    //  _                          _
                    // |  N_0(u[0])   ... N_n(u[0]) |
                    // |     ...      ...      ...  |
                    // |  N_0(u[m])   ... N_n(u[m]) |
                    //  -                          -
                    // TODO: N is going to be very sparse when it is large: switch to sparse representation
                    // N has semibandwidth < p  nonzero entries across diagonal
                    MatrixX<T> N = MatrixX<T>::Zero(m(k) + 1, n(k) + 1);    // coefficients matrix

                    for (int i = 0; i < N.rows(); i++)                      // the rows of N
                    {
                        // TODO: hard-coded for single tensor
                        int span = mfa_data.tmesh.FindSpan(k, input.params->param_grid[k][i], mfa_data.tmesh.tensor_prods[0]);
#ifndef MFA_TMESH       // original version for one tensor product
                        mfa_data.OrigBasisFuns(k, input.params->param_grid[k][i], span, N, i);
#else                   // tmesh version
                        mfa_data.BasisFuns(k, input.params->param_grid[k][i], span, N, i);
#endif
                    }

                    // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
                    // NtN has semibandwidth < p + 1 nonzero entries across diagonal
                    MatrixX<T> NtN = N.transpose() * N;

                    // R is the right hand side needed for solving NtN * P = R
                    MatrixX<T> R(N.cols(), pt_dim);

                    // P are the unknown interior control points and the solution to NtN * P = R
                    // NtN is positive definite -> do not need pivoting
                    // TODO: use a common representation for P and ctrl_pts to avoid copying
                    MatrixX<T> P(N.cols(), pt_dim);

                    size_t ncurves   = input.domain.rows() / input.ndom_pts(k);     // number of curves in this dimension
                    int nsame_steps  = 0;                                   // number of steps with same number of erroneous points
                    int n_step_sizes = 0;                                   // number of step sizes so far

                    // starting step size over curves
                    size_t s0 = ncurves / 2 > 0 ? ncurves / 2 : 1;

#ifdef MFA_CHECK_ALL_CURVES
                    s0              = 1;
                    max_num_curves  = ncurves;
#endif

                    for (size_t s = s0; s >= 1 && ncurves / s <= max_num_curves; s /= 2)     // for all step sizes over curves
                    {
                        bool new_max_nerr = false;                          // this step size changed the max_nerr

                        for (size_t j = 0; j < ncurves; j++)                // for all the curves in this dimension
                        {
                            // each time the step changes, shift start of s-th curves by one (by subtracting
                            // n_step-sizes below)
                            if (j >= n_step_sizes && (j - n_step_sizes) % s == 0)           // this is one of the s-th curves; compute it
                            {
                                // compute R from input domain points
                                RHS(k, N, R, weights, input.g.co[k][j]);

                                // rationalize NtN
                                MatrixX<T> NtN_rat = NtN;
                                mfa_data.Rationalize(k, weights, N, NtN_rat);

                                // solve for P
#ifdef WEIGH_ALL_DIMS                                                       // weigh all dimensions
                                P = NtN_rat.ldlt().solve(R);
#else                                                                       // don't weigh domain coordinate (only range)
                                // TODO: avoid 2 solves?
                                MatrixX<T> P2(P.rows(), P.cols());
                                P = NtN.ldlt().solve(R);                    // nonrational domain coordinates
                                P2 = NtN_rat.ldlt().solve(R);               // rational range coordinate
                                for (auto i = 0; i < P.rows(); i++)
                                    P(i, P.cols() - 1) = P2(i, P.cols() - 1);
#endif

                                // compute the error on the curve (number of input points with error > err_limit)
                                size_t nerr = ErrorCurve(k, t, input.g.co[k][j], P, weights, extents, err_spans, err_limit);

                                if (nerr > max_nerr)
                                {
                                    max_nerr     = nerr;
                                    new_max_nerr = true;
                                }
                            }
                        }                                                   // curves in this dimension

                        // stop refining step if no change
                        if (max_nerr && !new_max_nerr)
                            nsame_steps++;
                        if (nsame_steps == 2)
                            break;

                        n_step_sizes++;
                    }                                                       // step sizes over curves

                    // free R, NtN, and P
                    R.resize(0, 0);
                    NtN.resize(0, 0);
                    P.resize(0, 0);

                    // add new knots in the middle of spans with errors
                    new_knots[k].resize(err_spans.size());
                    new_levels[k].resize(new_knots[k].size());
                    tot_nnew_knots += new_knots[k].size();
                    size_t i = 0;                                           // index into new_knots
                    for (set<int>::iterator it = err_spans.begin(); it != err_spans.end(); ++it)
                    {
                        assert(*it < t.nctrl_pts[k]);                       // not trying to go beyond the last span

                        new_knots[k][i] = (mfa_data.tmesh.all_knots[k][*it] + mfa_data.tmesh.all_knots[k][*it + 1]) / 2.0;
                        new_levels[k][i] = t.level;
                        i++;
                    }

                    // print progress
                }                                                           // domain dimensions


                // insert the new knots
                mfa::NewKnots<T> nk(mfa_data, input);
                vector<vector<KnotIdx>> unused(dom_dim);
                nk.OrigInsertKnots(new_knots, new_levels, unused);

                // increase number of control points, weights, basis functions
                for (auto k = 0; k < dom_dim; k++)
                {
                    t.nctrl_pts(k) += new_knots[k].size();
                }
                auto tot_nctrl_pts = t.nctrl_pts.prod();
                t.ctrl_pts.resize(tot_nctrl_pts, t.ctrl_pts.cols());
                t.weights =  VectorX<T>::Ones(tot_nctrl_pts);
                mfa_data.tmesh.tensor_knot_idxs(t);
            }                                                               // tensor products

            return(tot_nnew_knots ? 0 : 1);
        }

    public:
        // Writes collocation matrix N to a file in triplet format (row-major order)
        void dumpCollocationMatrix(string name = "")
        {
            string filename = "N_encode" + name + ".txt";

            {
                fmt::print(" * Encoder: Dumping collocation matrix to {}\n", filename);
            }

            if (input.is_structured())
            {
                fmt::print("dumpCollocationMatrix is not yet supported for structured input.\n");
                return;
            }

            std::ofstream os;
            os.open(filename);

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[0];

            // control point strides
            VectorXi cps = VectorXi::Ones(dom_dim);
            for (int k = 1; k < dom_dim; k++)
            {
                cps(k) = cps(k-1) * t.nctrl_pts(k-1);
            }

            // Prep for evaluating basis functions
            VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
            BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
            vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
            for (int k = 0; k < dom_dim; k++)
            {
                lB[k].resize(q[k]);
            }

            VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
            VectorXi spans(dom_dim);                                       // current knot span in each dimension

            // Iterate thru every point in subvolume given by tensor
            VectorX<T> param(input.dom_dim);
            for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
            {
                input_it.params(param);

                // Compute basis functions at input point
                for (auto k = 0; k < dom_dim; k++)
                {
                    int p   = mfa_data.p(k);
                    T   u   = param(k);

                    spans[k] = mfa_data.tmesh.FindSpan(k, u);

                    ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

                    mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi);
                }

                // Compute matrix value and insert into Nt
                VolIterator ctrl_vol_iter(q);
                while (!ctrl_vol_iter.done())
                {
                    int ctrl_idx_full = 0;
                    T coeff_prod = 1;
                    for (int k = 0; k < dom_dim; k++)
                    {
                        int idx = ctrl_vol_iter.idx_dim(k);
                        ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
                        coeff_prod *= lB[k][idx];
                    }

                    if (coeff_prod != 0)
                    {
                        // Rows correspond to data, columns to control points
                        int row = input_it.idx();
                        int col = ctrl_idx_full;
                        fmt::print(os, "{} {} {}\n", row, col, coeff_prod);
                    }

                    ctrl_vol_iter.incr_iter();
                }
            }

            os.close();

            return;
        }
    };          // Encoder class

}           // mfa namespace


#endif
