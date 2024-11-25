#ifndef FUNCTIONS_AGI_H
#define FUNCTIONS_AGI_H

#include <cmath>
#include <vector>

#define N 32

namespace plb {

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Units conversion class
// A class for the conversion between meters-seconds to kilometers-minutes
///////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, template <typename NSU> class nsDescriptor>
class ConvertToKmMin {
public:
  ConvertToKmMin(T up_, T low_, T rhoUp_, T rhoLow_, T lx_, T ly_, T lz_,
                 T uCar_, T Di_, std::vector<T> &Dp_k_, std::vector<T> &rhoPi_,
                 std::vector<T> &rhoPa_, T g_, T mu_, T Ri_, T Gr_, T uMax_,
                 T resolution_ = T())
      : up(up_), low(low_), rhoUp(rhoUp_), rhoLow(rhoLow_), lx(lx_), ly(ly_),
        lz(lz_), uCar(uCar_), Di(Di_), Dp_k(Dp_k_), rhoPi(rhoPi_),
        rhoPa(rhoPa_), g(g_), mu(mu_), Ri(Ri_), Gr(Gr_), uMax(uMax_),
        resolution(resolution_) {}

  /// upper layer thickness
  T getUp() const { return up; }
  /// lower layer thickness
  T getLow() const { return low; }
  // upper layer density
  T getRhoUp() const { return rhoUp; }
  // lower layer density
  T getRhoLow() const { return rhoLow; }
  /// x-length
  T getLx() const { return lx; }
  /// y-length
  T getLy() const { return ly; }
  /// z-length
  T getLz() const { return lz; }
  /// characteristic velocity
  T getUcar() const { return uCar; }
  /// diffusivity
  T getDi() const { return Di; }
  /// particle diameter
  std::vector<T> getDp() const {
    std::vector<T> Dp_n;
    for (int i = 0; i < N; ++i)
      Dp_n.push_back(Dp_k[i]);
    return Dp_n;
  }
  /// particle density
  std::vector<T> getRhoPi() const {
    std::vector<T> rhoPi_n;
    for (int i = 0; i < N; ++i)
      rhoPi_n.push_back(rhoPi[i]);
    return rhoPi_n;
  }
  /// aggregate density
  std::vector<T> getRhoPa() const {
    std::vector<T> rhoPa_n;
    for (int i = 0; i < N; ++i)
      rhoPa_n.push_back(rhoPa[i]);
    return rhoPa_n;
  }
  /// gravitational acceleration
  T getG() const { return g; }
  /// dynamic viscosity
  T getMu() const { return mu; }
  /// Richardson number
  T getRi() const { return Ri; }
  /// Grashof number
  T getGr() const { return Gr; }
  // lattice spacing in dimensionless units
  T getDeltaX() const { return 1 / (T)resolution; }
  // temporal step
  T getDeltaT() const { return getLatticeU() * getDeltaX() / getUcar(); }
  /// conversion from dimensionless to lattice units for space coordinate
  plint nCell(T l) const { return (plint)(l / getDeltaX() + (T)0.5); }
  /// conversion from dimensionless to lattice units for time coordinuLbate
  plint nStep(T t) const { return (plint)(t / getDeltaT() + (T)0.5); }
  /// number of lattice cells in x-direction
  plint getNx() const { return nCell(getLx()) + 1; }
  /// number of lattice cells in y-direction
  plint getNy() const { return nCell(getLy()) + 1; }
  /// number of lattice cells in z-direction
  plint getNz() const { return nCell(getLz()) + 1; }
  /// velocity in lattice units (proportional to Mach number)
  T getLatticeU() const { return uMax; }
  /// Reynolds number
  T getRe() const { return std::sqrt(getGr() / getRi()); }
  /// viscosity in lattice units
  T getLatticeNu() const { return getLatticeU() * (getNz() - 1) / getRe(); }
  /// thermal conductivity in lattice units
  T getLatticeKappa() const {
    return (getDeltaT() / (getDeltaX() * getDeltaX()));
  }
  /// viscosity in lattice units
  T getLatticeGravity() const {
    return getDeltaT() * getDeltaT() / getDeltaX();
  }
  /// relaxation time
  T getSolventTau() const {
    return nsDescriptor<T>::invCs2 * getLatticeNu() + (T)0.5;
  }
  /// relaxation frequency
  T getSolventOmega() const { return (T)1 / getSolventTau(); }
  /// relaxation time
  /// relaxation frequency

private:
  T up, low, rhoUp, rhoLow, lx, ly, lz, uCar, Di;
  std::vector<T> &Dp_k, &rhoPi, &rhoPa;
  T g, mu, Ri, Gr, uMax, resolution;
};

template <typename T, template <typename NSU> class nsDescriptor>
void writeLogFile(ConvertToKmMin<T, nsDescriptor> const &parameters,
                  std::string const &title) {

  std::string fullName = global::directories().getLogOutDir() + "plbLog.dat";
  std::ofstream ofile(fullName.c_str());
  ofile << title << "\n\n";
  ofile << "Reynolds number:           Re=" << parameters.getRe() << "\n";
  ofile << "Upper layer density:           rhoUp=" << parameters.getRhoUp()
        << "\n";
  ofile << "Lower layer density:           rhoUp=" << parameters.getRhoLow()
        << "\n";
  //   ofile << "Particle density:           rhoPi=" << parameters.getRhoPi() <<
  //   "\n"; ofile << "Aggregate density:           rhoPa=" <<
  //   parameters.getRhoPa() << "\n";
  ofile << "Characteristic velocity:       uCar=" << parameters.getUcar()
        << "\n";
  ofile << "Gravitational acceleration:    g=" << parameters.getG() << "\n";
  ofile << "Dynamic viscosity:           mu=" << parameters.getMu() << "\n";
  ofile << "Richardson number:          Ri=" << parameters.getRi() << "\n";
  ofile << "Grasshoff number:            Gr=" << parameters.getGr() << "\n";
  ofile << "Kinematic viscosity:       Nu=" << parameters.getLatticeNu()
        << "\n";
  ofile << "Physical Diffusivity:      Di=" << parameters.getDi() << "\n";
  ofile << "Diffusivity:               Kappa=" << parameters.getLatticeKappa()
        << "\n";
  ofile << "Extent of the system:      lx=" << parameters.getLx() << "\n";
  ofile << "Extent of the system:      ly=" << parameters.getLy() << "\n";
  ofile << "Extent of the system:      lz=" << parameters.getLz() << "\n";
  ofile << "Grid spacing deltaX:       dx=" << parameters.getDeltaX() << "\n";
  ofile << "Time step deltaT:          dt=" << parameters.getDeltaT() << "\n";
  ofile << "Solvent omega:        omega_S=" << parameters.getSolventOmega()
        << "\n";
  ofile << "Caracteristic vel:       uLb=" << parameters.getLatticeU() << "\n";
  ofile << "Number of cells x:       Nx=" << parameters.getNx() << "\n";
  ofile << "Number of cells y:       Ny=" << parameters.getNy() << "\n";
  ofile << "Number of cells z:       Nz=" << parameters.getNz() << "\n";
  std::vector<T> Dp_n = parameters.getDp();
  for (std::size_t i = 0; i < Dp_n.size(); i++) {
    ofile << "Particle diameter:       Dp=" << Dp_n[i] << "\n";
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Buoyant force term
///////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, template <typename U> class FluidDescriptor>
class ScalarBuoyanTermProcessor3D_sev : public BoxProcessingFunctional3D {
public:
  ScalarBuoyanTermProcessor3D_sev(T gravity_, T rho0_, std::vector<T> rhoPi_,
                                  std::vector<T> rhoPa_, T dt_,
                                  Array<T, FluidDescriptor<T>::d> dir_);

  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> fields);
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual ScalarBuoyanTermProcessor3D_sev<T, FluidDescriptor> *clone() const;

private:
  T gravity, rho0;
  std::vector<T> rhoPi, rhoPa;
  T dt;
  Array<T, FluidDescriptor<T>::d> dir;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1st order upwind finite-difference scheme with neumann boundary conditions
// for density field
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class AdvectionDiffusionFd3D_neumann : public BoxProcessingFunctional3D {
public:
  AdvectionDiffusionFd3D_neumann(T d_, bool upwind_, bool neumann_, plint nx_,
                                 plint ny_, plint nz_);
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> fields);
  virtual AdvectionDiffusionFd3D_neumann<T> *clone() const;
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;

private:
  T d;
  bool upwind, neumann;
  plint nx, ny, nz;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Settling velocity field
///////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class Get_v_sedimentation : public BoxProcessingFunctional3D {
public:
  Get_v_sedimentation(std::vector<T> rhoP_, std::vector<T> Dp_, T convers_,
                      T mu_, T g_);
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> atomicBlocks);
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual Get_v_sedimentation<T> *clone() const;

private:
  std::vector<T> rhoP;
  std::vector<T> Dp;
  T convers;
  T mu;
  T g;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WENO3 procedure for the spatial derivative of the convective term using
// Lax-Friedrich flux splitting
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class AdvectionDiffusionFd3D2_WENO3_f : public BoxProcessingFunctional3D {
public:
  AdvectionDiffusionFd3D2_WENO3_f(T d_, T eps_, bool neumann_, plint nx_,
                                  plint ny_, plint nz_, Array<T, N> alpha_x_,
                                  Array<T, N> alpha_y_, Array<T, N> alpha_z_);
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> fields);
  virtual AdvectionDiffusionFd3D2_WENO3_f<T> *clone() const;
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;

private:
  T d, eps;
  bool neumann;
  plint nx, ny, nz;
  Array<T, N> alpha_x, alpha_y, alpha_z;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute fluxes
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, template <typename U> class FluidDescriptor>
class ComputeFluxes : public BoxProcessingFunctional3D {

public:
  ComputeFluxes(Array<T, N> alpha_x_, Array<T, N> alpha_y_,
                Array<T, N> alpha_z_);
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> fields);
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual ComputeFluxes<T, FluidDescriptor> *clone() const;

private:
  Array<T, N> alpha_x, alpha_y, alpha_z;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Third order Runge-Kutta for the temporal derivation (used with the WENO3 for
// the particle field)
///////////////////////////////////////////////////////////////////////////////////////////////////////////

/* ******** RK3_Step1_functional3D ****************************************** */

template <typename T>
class RK3_Step1_functional3D : public BoxProcessingFunctional3D {
public:
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> tensorFields);
  virtual RK3_Step1_functional3D<T> *clone() const;
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual BlockDomain::DomainT appliesTo() const;
};

/* ******** RK3_Step2_functional3D ****************************************** */

template <typename T>
class RK3_Step2_functional3D : public BoxProcessingFunctional3D {
public:
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> tensorFields);
  virtual RK3_Step2_functional3D<T> *clone() const;
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual BlockDomain::DomainT appliesTo() const;
};

/* ******** RK3_Step3_functional3D ****************************************** */

template <typename T>
class RK3_Step3_functional3D : public BoxProcessingFunctional3D {
public:
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> tensorFields);
  virtual RK3_Step3_functional3D<T> *clone() const;
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual BlockDomain::DomainT appliesTo() const;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Regularization function for the volfracField_tot
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class Regu_VF_functional3D : public BoxProcessingFunctional3D {
public:
  Regu_VF_functional3D();
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> fields);
  virtual Regu_VF_functional3D<T> *clone() const;
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual BlockDomain::DomainT appliesTo() const;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Source terms
///////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, template <typename U> class FluidDescriptor>
class BirthAndDeath : public BoxProcessingFunctional3D {
public:
  BirthAndDeath(const std::vector<std::vector<T>> &K_i_,
                const std::vector<std::vector<T>> &K_a_,
                const std::vector<std::vector<T>> &K_i_a_,
                const std::vector<std::vector<T>> &K_a_i_,
                const std::vector<std::vector<T>> &KTS_i_,
                const std::vector<std::vector<T>> &KTS_a_,
                const std::vector<std::vector<T>> &KTS_i_a_, T dx_, T dt_,
                const std::vector<T> &Dp_, const std::vector<T> &Dp_agg_,
                const std::vector<T> &m_,
                const std::vector<std::vector<std::vector<T>>> &coeff_,
                const std::vector<T> rhoPi_, const std::vector<T> rhoPa_);
  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> fields);
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual BirthAndDeath<T, FluidDescriptor> *clone() const;

private:
  const std::vector<std::vector<T>> &K_i, &K_a, &K_i_a, &K_a_i, &KTS_i, &KTS_a,
      &KTS_i_a;
  T dx, dt;
  const std::vector<T> &Dp, &Dp_agg, &m;
  const std::vector<std::vector<std::vector<T>>> &coeff;
  const std::vector<T> rhoPi, rhoPa;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dissipation rate
///////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, template <typename U> class FluidDescriptor>
class ComputeDissipationRate : public BoxProcessingFunctional3D {
public:
  ComputeDissipationRate(T cSmago_);

  virtual void processGenericBlocks(Box3D domain,
                                    std::vector<AtomicBlock3D *> fields);
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const;
  virtual ComputeDissipationRate<T, FluidDescriptor> *clone() const;

private:
  T cSmago;
};

} // namespace plb
#endif
