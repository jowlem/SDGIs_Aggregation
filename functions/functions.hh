#ifndef FUNCTIONS_AGI_HH
#define FUNCTIONS_AGI_HH

#include "functions.h"
#include <cmath>
#include <vector>

#define M_PI 3.14159265358979323846

namespace plb {

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Buoyant force term
///////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, template <typename U> class FluidDescriptor>
ScalarBuoyanTermProcessor3D_sev<T, FluidDescriptor>::
    ScalarBuoyanTermProcessor3D_sev(T gravity_, T rho0_, std::vector<T> rhoPi_,
                                    std::vector<T> rhoPa_, T dt_,
                                    Array<T, FluidDescriptor<T>::d> dir_)
    : gravity(gravity_), rho0(rho0_), rhoPi(rhoPi_), rhoPa(rhoPa_), dt(dt_),
      dir(dir_) {
  // We normalize the direction of the force vector.
  T normDir = std::sqrt(VectorTemplate<T, FluidDescriptor>::normSqr(dir));
  for (pluint iD = 0; iD < FluidDescriptor<T>::d; ++iD) {
    dir[iD] /= normDir;
  }
}

template <typename T, template <typename U> class FluidDescriptor>
void ScalarBuoyanTermProcessor3D_sev<T, FluidDescriptor>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {
  typedef FluidDescriptor<T> D;
  enum { forceOffset = FluidDescriptor<T>::ExternalField::forceBeginsAt };

  PLB_PRECONDITION(fields.size() == 4);
  BlockLattice3D<T, FluidDescriptor> *fluid =
      dynamic_cast<BlockLattice3D<T, FluidDescriptor> *>(fields[0]);
  ScalarField3D<T> *densityfield = dynamic_cast<ScalarField3D<T> *>(fields[1]);
  TensorField3D<T, N> *volfracfield =
      dynamic_cast<TensorField3D<T, N> *>(fields[2]);
  TensorField3D<T, N> *volfracfield_agg =
      dynamic_cast<TensorField3D<T, N> *>(fields[3]);

  Dot3D offset1 = computeRelativeDisplacement(*fluid, *densityfield);
  Dot3D offset2 = computeRelativeDisplacement(*fluid, *volfracfield);
  Dot3D offset3 = computeRelativeDisplacement(*fluid, *volfracfield_agg);

  Array<T, D::d> gravOverrho0(gravity * dir[0] / rho0, gravity * dir[1] / rho0,
                              gravity * dir[2] / rho0);

  T maxiT = 1.0 / dt;
  T localVolfrac_i_tot;
  T localVolfrac_a_tot;
  T force_i, force_a;
  T iT = fluid->getTimeCounter().getTime();
  T gain = util::sinIncreasingFunction(iT, maxiT);
  T diffT;
  T diffT_a;

  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {

        Array<T, N> localVolfrac =
            volfracfield->get(iX + offset2.x, iY + offset2.y, iZ + offset2.z);
        Array<T, N> localVolfrac_agg = volfracfield_agg->get(
            iX + offset3.x, iY + offset3.y, iZ + offset3.z);
        // Computation of the Boussinesq force
        T *force = fluid->get(iX, iY, iZ).getExternal(forceOffset);
        T dens =
            densityfield->get(iX + offset1.x, iY + offset1.y, iZ + offset1.z);
        for (pluint id = 0; id < D::d; ++id) {
          localVolfrac_i_tot = 0.;
          localVolfrac_a_tot = 0.;
          diffT = 0.;
          diffT_a = 0.;
          for (plint i = 0; i < N; ++i) {
            diffT += (rhoPi[i] - rho0) * localVolfrac[i];
            diffT_a += (rhoPa[i] - rho0) * localVolfrac_agg[i];
            localVolfrac_i_tot += localVolfrac[i];
            localVolfrac_a_tot += localVolfrac_agg[i];
          }
          force_i = diffT - (dens - rho0) * localVolfrac_i_tot;
          force_a = diffT_a - (dens - rho0) * localVolfrac_a_tot;
          force[id] =
              -gain * gravOverrho0[id] * (force_i + force_a + (dens - rho0));
        }
      }
    }
  }
}

template <typename T, template <typename U> class FluidDescriptor>
ScalarBuoyanTermProcessor3D_sev<T, FluidDescriptor> *
ScalarBuoyanTermProcessor3D_sev<T, FluidDescriptor>::clone() const {
  return new ScalarBuoyanTermProcessor3D_sev<T, FluidDescriptor>(*this);
}

template <typename T, template <typename U> class FluidDescriptor>
void ScalarBuoyanTermProcessor3D_sev<T, FluidDescriptor>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::staticVariables;
  modified[1] = modif::nothing;
  modified[2] = modif::nothing;
  modified[3] = modif::nothing;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1st order upwind finite-difference scheme with neumann boundary conditions
// for density field
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
AdvectionDiffusionFd3D_neumann<T>::AdvectionDiffusionFd3D_neumann(
    T d_, bool upwind_, bool neumann_, plint nx_, plint ny_, plint nz_)
    : d(d_), upwind(upwind_), neumann(neumann_), nx(nx_), ny(ny_), nz(nz_) {}

template <typename T>
void AdvectionDiffusionFd3D_neumann<T>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {
  PLB_PRECONDITION(fields.size() == 5);
  ScalarField3D<T> *phi_t = dynamic_cast<ScalarField3D<T> *>(fields[0]);
  ScalarField3D<T> *phi_tp1 = dynamic_cast<ScalarField3D<T> *>(fields[1]);
  ScalarField3D<T> *result = dynamic_cast<ScalarField3D<T> *>(fields[2]);
  TensorField3D<T, 3> *uField = dynamic_cast<TensorField3D<T, 3> *>(fields[3]);
  ScalarField3D<T> *Q = dynamic_cast<ScalarField3D<T> *>(fields[4]);

  Dot3D ofs1 = computeRelativeDisplacement(*phi_t, *phi_tp1);
  Dot3D ofs2 = computeRelativeDisplacement(*phi_t, *result);
  Dot3D ofs3 = computeRelativeDisplacement(*phi_t, *uField);
  Dot3D ofs4 = computeRelativeDisplacement(*phi_t, *Q);

  Dot3D absoluteOffset = phi_tp1->getLocation();

  if (upwind) {
    for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
      for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
        for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {

          plint absoluteZ = absoluteOffset.z + iZ + ofs1.z;

          T phiC = phi_tp1->get(iX + ofs1.x, iY + ofs1.y, iZ + ofs1.z);
          T phiE = phi_tp1->get(iX + 1 + ofs1.x, iY + ofs1.y, iZ + ofs1.z);
          T phiW = phi_tp1->get(iX - 1 + ofs1.x, iY + ofs1.y, iZ + ofs1.z);
          T phiN = phi_tp1->get(iX + ofs1.x, iY + 1 + ofs1.y, iZ + ofs1.z);
          T phiS = phi_tp1->get(iX + ofs1.x, iY - 1 + ofs1.y, iZ + ofs1.z);
          T phiT = phi_tp1->get(iX + ofs1.x, iY + ofs1.y, iZ + 1 + ofs1.z);
          T phiB = phi_tp1->get(iX + ofs1.x, iY + ofs1.y, iZ - 1 + ofs1.z);

          Array<T, 3> const &u =
              uField->get(iX + ofs3.x, iY + ofs3.y, iZ + ofs3.z);

          Array<T, 3> adv;
          T diffX, diffY, diffZ;

          adv[0] =
              (util::greaterThan(u[0], (T)0)
                   ? (phiC - phiW)
                   : (util::lessThan(u[0], (T)0) ? (phiE - phiC)
                                                 : (T)0.5 * (phiE - phiW)));
          adv[1] =
              (util::greaterThan(u[1], (T)0)
                   ? (phiC - phiS)
                   : (util::lessThan(u[1], (T)0) ? (phiN - phiC)
                                                 : (T)0.5 * (phiN - phiS)));
          adv[2] =
              (util::greaterThan(u[2], (T)0)
                   ? (phiC - phiB)
                   : (util::lessThan(u[2], (T)0) ? (phiT - phiC)
                                                 : (T)0.5 * (phiT - phiB)));

          diffX = phiW + phiE - (T)2 * phiC;
          diffY = phiS + phiN - (T)2 * phiC;
          diffZ = phiT + phiB - (T)2 * phiC;

          if (neumann) {
            if (absoluteZ == 0) {
              adv[2] = 0;
              diffZ = (T)2 * phiT - (T)2 * phiC;
            }

            if (absoluteZ == nz - 1) {
              adv[2] = 0;
              diffZ = (T)2 * phiB - (T)2 * phiC;
            }
          }

          T advection = u[0] * adv[0] + u[1] * adv[1] + u[2] * adv[2];

          T diffusion = d * (diffX + diffY + diffZ);

          result->get(iX + ofs2.x, iY + ofs2.y, iZ + ofs2.z) =
              phi_t->get(iX, iY, iZ) + diffusion - advection +
              Q->get(iX + ofs4.x, iY + ofs4.y, iZ + ofs4.z);
        }
      }
    }
  } else {
    for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
      for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
        for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {

          T phiC = phi_tp1->get(iX + ofs1.x, iY + ofs1.y, iZ + ofs1.z);
          T phiE = phi_tp1->get(iX + 1 + ofs1.x, iY + ofs1.y, iZ + ofs1.z);
          T phiW = phi_tp1->get(iX - 1 + ofs1.x, iY + ofs1.y, iZ + ofs1.z);
          T phiN = phi_tp1->get(iX + ofs1.x, iY + 1 + ofs1.y, iZ + ofs1.z);
          T phiS = phi_tp1->get(iX + ofs1.x, iY - 1 + ofs1.y, iZ + ofs1.z);
          T phiT = phi_tp1->get(iX + ofs1.x, iY + ofs1.y, iZ + 1 + ofs1.z);
          T phiB = phi_tp1->get(iX + ofs1.x, iY + ofs1.y, iZ - 1 + ofs1.z);

          Array<T, 3> const &u =
              uField->get(iX + ofs3.x, iY + ofs3.y, iZ + ofs3.z);

          T advection = (T)0.5 * (u[0] * (phiE - phiW) + u[1] * (phiN - phiS) +
                                  u[2] * (phiT - phiB));

          T diffusion =
              d * (phiE + phiW + phiN + phiS + phiT + phiB - (T)6 * phiC);

          result->get(iX + ofs2.x, iY + ofs2.y, iZ + ofs2.z) =
              phi_t->get(iX, iY, iZ) + diffusion - advection +
              Q->get(iX + ofs4.x, iY + ofs4.y, iZ + ofs4.z);
        }
      }
    }
  }
}

template <typename T>
AdvectionDiffusionFd3D_neumann<T> *
AdvectionDiffusionFd3D_neumann<T>::clone() const {
  return new AdvectionDiffusionFd3D_neumann<T>(*this);
}

template <typename T>
void AdvectionDiffusionFd3D_neumann<T>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;         // phi_t
  modified[1] = modif::nothing;         // phi_tp1
  modified[2] = modif::staticVariables; // result
  modified[3] = modif::nothing;         // u
  modified[4] = modif::nothing;         // Q
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Settling velocity field
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
Get_v_sedimentation<T>::Get_v_sedimentation(std::vector<T> rhoP_,
                                            std::vector<T> Dp_, T convers_,
                                            T mu_, T g_)
    : rhoP(rhoP_), Dp(Dp_), convers(convers_), mu(mu_), g(g_) {}

template <typename T>
void Get_v_sedimentation<T>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> atomicBlocks) {
  // typedef DensityDescriptor<T> D;
  PLB_PRECONDITION(atomicBlocks.size() == 2);
  ScalarField3D<T> *densityField =
      dynamic_cast<ScalarField3D<T> *>(atomicBlocks[0]);
  TensorField3D<T, N> *v_sed =
      dynamic_cast<TensorField3D<T, N> *>(atomicBlocks[1]);

  Dot3D offset = computeRelativeDisplacement(*densityField, *v_sed);

  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {

        T dens = densityField->get(iX, iY, iZ);
        Array<T, N> vel_sed;

        for (pluint iD = 0; iD < N; ++iD) {
          vel_sed[iD] = -convers *
                        (0.5 * Dp[iD] * Dp[iD] * g * (rhoP[iD] - dens)) /
                        (9 * mu);
        }
        v_sed->get(iX + offset.x, iY + offset.y, iZ + offset.z) = vel_sed;
      }
    }
  }
}

template <typename T>
Get_v_sedimentation<T> *Get_v_sedimentation<T>::clone() const {
  return new Get_v_sedimentation<T>(*this);
}

template <typename T>
void Get_v_sedimentation<T>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;
  modified[1] = modif::staticVariables;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WENO3 procedure for the spatial derivative of the convective term using
// Lax-Friedrich flux splitting
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
AdvectionDiffusionFd3D2_WENO3_f<T>::AdvectionDiffusionFd3D2_WENO3_f(
    T d_, T eps_, bool neumann_, plint nx_, plint ny_, plint nz_,
    Array<T, N> alpha_x_, Array<T, N> alpha_y_, Array<T, N> alpha_z_)
    : d(d_), eps(eps_), neumann(neumann_), nx(nx_), ny(ny_), nz(nz_),
      alpha_x(alpha_x_), alpha_y(alpha_y_), alpha_z(alpha_z_) {}

template <typename T>
void AdvectionDiffusionFd3D2_WENO3_f<T>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {
  PLB_PRECONDITION(fields.size() == 8);
  TensorField3D<T, N> *Flux_px = dynamic_cast<TensorField3D<T, N> *>(fields[0]);
  TensorField3D<T, N> *Flux_py = dynamic_cast<TensorField3D<T, N> *>(fields[1]);
  TensorField3D<T, N> *Flux_pz = dynamic_cast<TensorField3D<T, N> *>(fields[2]);
  TensorField3D<T, N> *Flux_nx = dynamic_cast<TensorField3D<T, N> *>(fields[3]);
  TensorField3D<T, N> *Flux_ny = dynamic_cast<TensorField3D<T, N> *>(fields[4]);
  TensorField3D<T, N> *Flux_nz = dynamic_cast<TensorField3D<T, N> *>(fields[5]);
  TensorField3D<T, N> *result = dynamic_cast<TensorField3D<T, N> *>(fields[6]);
  TensorField3D<T, N> *Q = dynamic_cast<TensorField3D<T, N> *>(fields[7]);

  Dot3D ofs1 = computeRelativeDisplacement(*Flux_px, *Flux_py);
  Dot3D ofs2 = computeRelativeDisplacement(*Flux_px, *Flux_pz);
  Dot3D ofs3 = computeRelativeDisplacement(*Flux_px, *Flux_nx);
  Dot3D ofs4 = computeRelativeDisplacement(*Flux_px, *Flux_ny);
  Dot3D ofs5 = computeRelativeDisplacement(*Flux_px, *Flux_nz);
  Dot3D ofs6 = computeRelativeDisplacement(*Flux_px, *result);
  Dot3D ofs7 = computeRelativeDisplacement(*Flux_px, *Q);

  Dot3D absoluteOffset = Flux_px->getLocation();

  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {

        plint absoluteZ = absoluteOffset.z + iZ + ofs1.z;

        Array<T, N> res;

        for (plint iD = 0; iD < N; ++iD) {

          // x-axis
          // /////////////////////////////////////////////////////////////////
          T fp_i = Flux_px->get(iX, iY, iZ)[iD];
          T fp_ip1 = Flux_px->get(iX + 1, iY, iZ)[iD];
          T fp_im1 = Flux_px->get(iX - 1, iY, iZ)[iD];
          T fp_im2 = Flux_px->get(iX - 2, iY, iZ)[iD];
          T fn_i = Flux_nx->get(iX + ofs3.x, iY + ofs3.y, iZ + ofs3.z)[iD];
          T fn_ip1 =
              Flux_nx->get(iX + 1 + ofs3.x, iY + ofs3.y, iZ + ofs3.z)[iD];
          T fn_im1 =
              Flux_nx->get(iX - 1 + ofs3.x, iY + ofs3.y, iZ + ofs3.z)[iD];
          T fn_ip2 =
              Flux_nx->get(iX + 2 + ofs3.x, iY + ofs3.y, iZ + ofs3.z)[iD];

          T betaX1 = (fp_ip1 - fp_i) * (fp_ip1 - fp_i);
          T betaX2 = (fp_i - fp_im1) * (fp_i - fp_im1);
          T betaX3 = (fn_ip2 - fn_ip1) * (fn_ip2 - fn_ip1);
          T betaX4 = (fn_ip1 - fn_i) * (fn_ip1 - fn_i);
          T betaX5 = (fp_i - fp_im1) * (fp_i - fp_im1);
          T betaX6 = (fp_im1 - fp_im2) * (fp_im1 - fp_im2);
          T betaX7 = (fn_ip1 - fn_i) * (fn_ip1 - fn_i);
          T betaX8 = (fn_i - fn_im1) * (fn_i - fn_im1);

          T wX1_t = (T)2 / ((T)3 * (eps + betaX1) * (eps + betaX1));
          T wX2_t = (T)1 / ((T)3 * (eps + betaX2) * (eps + betaX2));
          T wX3_t = (T)2 / ((T)3 * (eps + betaX3) * (eps + betaX3));
          T wX4_t = (T)1 / ((T)3 * (eps + betaX4) * (eps + betaX4));
          T wX5_t = (T)2 / ((T)3 * (eps + betaX5) * (eps + betaX5));
          T wX6_t = (T)1 / ((T)3 * (eps + betaX6) * (eps + betaX6));
          T wX7_t = (T)2 / ((T)3 * (eps + betaX7) * (eps + betaX7));
          T wX8_t = (T)1 / ((T)3 * (eps + betaX8) * (eps + betaX8));

          T wX1 = wX1_t / (wX1_t + wX2_t);
          T wX2 = wX2_t / (wX1_t + wX2_t);
          T wX3 = wX3_t / (wX3_t + wX4_t);
          T wX4 = wX4_t / (wX3_t + wX4_t);
          T wX5 = wX5_t / (wX5_t + wX6_t);
          T wX6 = wX6_t / (wX5_t + wX6_t);
          T wX7 = wX7_t / (wX7_t + wX8_t);
          T wX8 = wX8_t / (wX7_t + wX8_t);

          T fp_p12 = wX1 * (((T)1 / 2) * fp_i + ((T)1 / 2) * fp_ip1) +
                     wX2 * (((T)3 / 2) * fp_i - ((T)1 / 2) * fp_im1);
          T fn_p12 = wX3 * (((T)3 / 2) * fn_ip1 - ((T)1 / 2) * fn_ip2) +
                     wX4 * (((T)1 / 2) * fn_i + ((T)1 / 2) * fn_ip1);
          T fp_m12 = wX5 * (((T)1 / 2) * fp_im1 + ((T)1 / 2) * fp_i) +
                     wX6 * (((T)3 / 2) * fp_im1 - ((T)1 / 2) * fp_im2);
          T fn_m12 = wX7 * (((T)3 / 2) * fn_i - ((T)1 / 2) * fn_ip1) +
                     wX8 * (((T)1 / 2) * fn_im1 + ((T)1 / 2) * fn_i);

          T f_p12 = fp_p12 + fn_p12;
          T f_m12 = fp_m12 + fn_m12;

          // y-axis
          // ////////////////////////////////////////////////////////////////////
          T gp_i = Flux_py->get(iX + ofs1.x, iY + ofs1.y, iZ + ofs1.z)[iD];
          T gp_ip1 =
              Flux_py->get(iX + ofs1.x, iY + 1 + ofs1.y, iZ + ofs1.z)[iD];
          T gp_im1 =
              Flux_py->get(iX + ofs1.x, iY - 1 + ofs1.y, iZ + ofs1.z)[iD];
          T gp_im2 =
              Flux_py->get(iX + ofs1.x, iY - 2 + ofs1.y, iZ + ofs1.z)[iD];
          T gn_i = Flux_ny->get(iX + ofs4.x, iY + ofs4.y, iZ + ofs4.z)[iD];
          T gn_ip1 =
              Flux_ny->get(iX + ofs4.x, iY + 1 + ofs4.y, iZ + ofs4.z)[iD];
          T gn_im1 =
              Flux_ny->get(iX + ofs4.x, iY - 1 + ofs4.y, iZ + ofs4.z)[iD];
          T gn_ip2 =
              Flux_ny->get(iX + ofs4.x, iY + 2 + ofs4.y, iZ + ofs4.z)[iD];

          T betaY1 = (gp_ip1 - gp_i) * (gp_ip1 - gp_i);
          T betaY2 = (gp_i - gp_im1) * (gp_i - gp_im1);
          T betaY3 = (gn_ip2 - gn_ip1) * (gn_ip2 - gn_ip1);
          T betaY4 = (gn_ip1 - gn_i) * (gn_ip1 - gn_i);
          T betaY5 = (gp_i - gp_im1) * (gp_i - gp_im1);
          T betaY6 = (gp_im1 - gp_im2) * (gp_im1 - gp_im2);
          T betaY7 = (gn_ip1 - gn_i) * (gn_ip1 - gn_i);
          T betaY8 = (gn_i - gn_im1) * (gn_i - gn_im1);

          T wY1_t = (T)2 / ((T)3 * (eps + betaY1) * (eps + betaY1));
          T wY2_t = (T)1 / ((T)3 * (eps + betaY2) * (eps + betaY2));
          T wY3_t = (T)2 / ((T)3 * (eps + betaY3) * (eps + betaY3));
          T wY4_t = (T)1 / ((T)3 * (eps + betaY4) * (eps + betaY4));
          T wY5_t = (T)2 / ((T)3 * (eps + betaY5) * (eps + betaY5));
          T wY6_t = (T)1 / ((T)3 * (eps + betaY6) * (eps + betaY6));
          T wY7_t = (T)2 / ((T)3 * (eps + betaY7) * (eps + betaY7));
          T wY8_t = (T)1 / ((T)3 * (eps + betaY8) * (eps + betaY8));

          T wY1 = wY1_t / (wY1_t + wY2_t);
          T wY2 = wY2_t / (wY1_t + wY2_t);
          T wY3 = wY3_t / (wY3_t + wY4_t);
          T wY4 = wY4_t / (wY3_t + wY4_t);
          T wY5 = wY5_t / (wY5_t + wY6_t);
          T wY6 = wY6_t / (wY5_t + wY6_t);
          T wY7 = wY7_t / (wY7_t + wY8_t);
          T wY8 = wY8_t / (wY7_t + wY8_t);

          T gp_p12 = wY1 * (((T)1 / 2) * gp_i + ((T)1 / 2) * gp_ip1) +
                     wY2 * (((T)3 / 2) * gp_i - ((T)1 / 2) * gp_im1);
          T gn_p12 = wY3 * (((T)3 / 2) * gn_ip1 - ((T)1 / 2) * gn_ip2) +
                     wY4 * (((T)1 / 2) * gn_i + ((T)1 / 2) * gn_ip1);
          T gp_m12 = wY5 * (((T)1 / 2) * gp_im1 + ((T)1 / 2) * gp_i) +
                     wY6 * (((T)3 / 2) * gp_im1 - ((T)1 / 2) * gp_im2);
          T gn_m12 = wY7 * (((T)3 / 2) * gn_i - ((T)1 / 2) * gn_ip1) +
                     wY8 * (((T)1 / 2) * gn_im1 + ((T)1 / 2) * gn_i);

          T g_p12 = gp_p12 + gn_p12;
          T g_m12 = gp_m12 + gn_m12;

          // z-axis
          // ////////////////////////////////////////////////////////////////////
          T hp_i = Flux_pz->get(iX + ofs2.x, iY + ofs2.y, iZ + ofs2.z)[iD];
          T hp_ip1 =
              Flux_pz->get(iX + ofs2.x, iY + ofs2.y, iZ + 1 + ofs2.z)[iD];
          T hp_im1 =
              Flux_pz->get(iX + ofs2.x, iY + ofs2.y, iZ - 1 + ofs2.z)[iD];
          T hp_im2 =
              Flux_pz->get(iX + ofs2.x, iY + ofs2.y, iZ - 2 + ofs2.z)[iD];
          T hn_i = Flux_nz->get(iX + ofs5.x, iY + ofs5.y, iZ + ofs5.z)[iD];
          T hn_ip1 =
              Flux_nz->get(iX + ofs5.x, iY + ofs5.y, iZ + 1 + ofs5.z)[iD];
          T hn_im1 =
              Flux_nz->get(iX + ofs5.x, iY + ofs5.y, iZ - 1 + ofs5.z)[iD];
          T hn_ip2 =
              Flux_nz->get(iX + ofs5.x, iY + ofs5.y, iZ + 2 + ofs5.z)[iD];

          T betaZ1 = (hp_ip1 - hp_i) * (hp_ip1 - hp_i);
          T betaZ2 = (hp_i - hp_im1) * (hp_i - hp_im1);
          T betaZ3 = (hn_ip2 - hn_ip1) * (hn_ip2 - hn_ip1);
          T betaZ4 = (hn_ip1 - hn_i) * (hn_ip1 - hn_i);
          T betaZ5 = (hp_i - hp_im1) * (hp_i - hp_im1);
          T betaZ6 = (hp_im1 - hp_im2) * (hp_im1 - hp_im2);
          T betaZ7 = (hn_ip1 - hn_i) * (hn_ip1 - hn_i);
          T betaZ8 = (hn_i - hn_im1) * (hn_i - hn_im1);

          T wZ1_t = (T)2 / ((T)3 * (eps + betaZ1) * (eps + betaZ1));
          T wZ2_t = (T)1 / ((T)3 * (eps + betaZ2) * (eps + betaZ2));
          T wZ3_t = (T)2 / ((T)3 * (eps + betaZ3) * (eps + betaZ3));
          T wZ4_t = (T)1 / ((T)3 * (eps + betaZ4) * (eps + betaZ4));
          T wZ5_t = (T)2 / ((T)3 * (eps + betaZ5) * (eps + betaZ5));
          T wZ6_t = (T)1 / ((T)3 * (eps + betaZ6) * (eps + betaZ6));
          T wZ7_t = (T)2 / ((T)3 * (eps + betaZ7) * (eps + betaZ7));
          T wZ8_t = (T)1 / ((T)3 * (eps + betaZ8) * (eps + betaZ8));

          T wZ1 = wZ1_t / (wZ1_t + wZ2_t);
          T wZ2 = wZ2_t / (wZ1_t + wZ2_t);
          T wZ3 = wZ3_t / (wZ3_t + wZ4_t);
          T wZ4 = wZ4_t / (wZ3_t + wZ4_t);
          T wZ5 = wZ5_t / (wZ5_t + wZ6_t);
          T wZ6 = wZ6_t / (wZ5_t + wZ6_t);
          T wZ7 = wZ7_t / (wZ7_t + wZ8_t);
          T wZ8 = wZ8_t / (wZ7_t + wZ8_t);

          T hp_p12 = wZ1 * (((T)1 / 2) * hp_i + ((T)1 / 2) * hp_ip1) +
                     wZ2 * (((T)3 / 2) * hp_i - ((T)1 / 2) * hp_im1);
          T hn_p12 = wZ3 * (((T)3 / 2) * hn_ip1 - ((T)1 / 2) * hn_ip2) +
                     wZ4 * (((T)1 / 2) * hn_i + ((T)1 / 2) * hn_ip1);
          T hp_m12 = wZ5 * (((T)1 / 2) * hp_im1 + ((T)1 / 2) * hp_i) +
                     wZ6 * (((T)3 / 2) * hp_im1 - ((T)1 / 2) * hp_im2);
          T hn_m12 = wZ7 * (((T)3 / 2) * hn_i - ((T)1 / 2) * hn_ip1) +
                     wZ8 * (((T)1 / 2) * hn_im1 + ((T)1 / 2) * hn_i);

          T h_p12 = hp_p12 + hn_p12;
          T h_m12 = hp_m12 + hn_m12;

          Array<T, 3> adv;
          adv[0] = (f_p12 - f_m12);
          adv[1] = (g_p12 - g_m12);
          adv[2] = (h_p12 - h_m12);

          if (neumann) {

            if (absoluteZ == 0) {
              adv[2] = h_p12;
            }
            if (absoluteZ == nz - 1) {
              adv[2] = -h_m12;
            }
          }
          T advection = adv[0] + adv[1] + adv[2];

          res[iD] = -advection;
        }

        result->get(iX + ofs6.x, iY + ofs6.y, iZ + ofs6.z) =
            res + Q->get(iX + ofs7.x, iY + ofs7.y, iZ + ofs7.z);
      }
    }
  }
}

template <typename T>
AdvectionDiffusionFd3D2_WENO3_f<T> *
AdvectionDiffusionFd3D2_WENO3_f<T>::clone() const {
  return new AdvectionDiffusionFd3D2_WENO3_f<T>(*this);
}

template <typename T>
void AdvectionDiffusionFd3D2_WENO3_f<T>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;
  modified[1] = modif::nothing;
  modified[2] = modif::nothing;
  modified[3] = modif::nothing;
  modified[4] = modif::nothing;
  modified[5] = modif::nothing;
  modified[6] = modif::staticVariables;
  modified[7] = modif::nothing;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute fluxes
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, template <typename U> class FluidDescriptor>
ComputeFluxes<T, FluidDescriptor>::ComputeFluxes(Array<T, N> alpha_x_,
                                                 Array<T, N> alpha_y_,
                                                 Array<T, N> alpha_z_)
    : alpha_x(alpha_x_), alpha_y(alpha_y_), alpha_z(alpha_z_) {}

template <typename T, template <typename U> class FluidDescriptor>
void ComputeFluxes<T, FluidDescriptor>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {
  PLB_PRECONDITION(fields.size() == 9);
  TensorField3D<T, N> *volfracField =
      dynamic_cast<TensorField3D<T, N> *>(fields[0]);
  TensorField3D<T, N> *Flux_px = dynamic_cast<TensorField3D<T, N> *>(fields[1]);
  TensorField3D<T, N> *Flux_py = dynamic_cast<TensorField3D<T, N> *>(fields[2]);
  TensorField3D<T, N> *Flux_pz = dynamic_cast<TensorField3D<T, N> *>(fields[3]);
  TensorField3D<T, N> *Flux_nx = dynamic_cast<TensorField3D<T, N> *>(fields[4]);
  TensorField3D<T, N> *Flux_ny = dynamic_cast<TensorField3D<T, N> *>(fields[5]);
  TensorField3D<T, N> *Flux_nz = dynamic_cast<TensorField3D<T, N> *>(fields[6]);
  BlockLattice3D<T, FluidDescriptor> *fluid =
      dynamic_cast<BlockLattice3D<T, FluidDescriptor> *>(fields[7]);
  TensorField3D<T, N> *v_sedimentation =
      dynamic_cast<TensorField3D<T, N> *>(fields[8]);

  Dot3D ofs1 = computeRelativeDisplacement(*volfracField, *Flux_px);
  Dot3D ofs2 = computeRelativeDisplacement(*volfracField, *Flux_py);
  Dot3D ofs3 = computeRelativeDisplacement(*volfracField, *Flux_pz);
  Dot3D ofs4 = computeRelativeDisplacement(*volfracField, *Flux_nx);
  Dot3D ofs5 = computeRelativeDisplacement(*volfracField, *Flux_ny);
  Dot3D ofs6 = computeRelativeDisplacement(*volfracField, *Flux_nz);
  Dot3D ofs7 = computeRelativeDisplacement(*volfracField, *fluid);
  Dot3D ofs8 = computeRelativeDisplacement(*volfracField, *v_sedimentation);

  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
        Array<T, 3> u;
        Array<T, N> f_px, f_py, f_pz, f_nx, f_ny, f_nz;
        Array<T, N> phi = volfracField->get(iX, iY, iZ);
        Array<T, N> vsed =
            v_sedimentation->get(iX + ofs8.x, iY + ofs8.y, iZ + ofs8.z);
        fluid->get(iX + ofs7.x, iY + ofs7.y, iZ + ofs7.z).computeVelocity(u);

        for (plint iD = 0; iD < N; ++iD) {

          f_px[iD] = ((T)1 / (T)2) * (u[0] * phi[iD] + alpha_x[iD] * phi[iD]);
          f_py[iD] = ((T)1 / (T)2) * (u[1] * phi[iD] + alpha_y[iD] * phi[iD]);
          f_pz[iD] = ((T)1 / (T)2) *
                     ((u[2] + vsed[iD]) * phi[iD] + alpha_z[iD] * phi[iD]);
          f_nx[iD] = ((T)1 / (T)2) * (u[0] * phi[iD] - alpha_x[iD] * phi[iD]);
          f_ny[iD] = ((T)1 / (T)2) * (u[1] * phi[iD] - alpha_y[iD] * phi[iD]);
          f_nz[iD] = ((T)1 / (T)2) *
                     ((u[2] + vsed[iD]) * phi[iD] - alpha_z[iD] * phi[iD]);
        }

        Flux_px->get(iX + ofs1.x, iY + ofs1.y, iZ + ofs1.z) = f_px;
        Flux_py->get(iX + ofs2.x, iY + ofs2.y, iZ + ofs2.z) = f_py;
        Flux_pz->get(iX + ofs3.x, iY + ofs3.y, iZ + ofs3.z) = f_pz;
        Flux_nx->get(iX + ofs4.x, iY + ofs4.y, iZ + ofs4.z) = f_nx;
        Flux_ny->get(iX + ofs5.x, iY + ofs5.y, iZ + ofs5.z) = f_ny;
        Flux_nz->get(iX + ofs6.x, iY + ofs6.y, iZ + ofs6.z) = f_nz;
      }
    }
  }
}

template <typename T, template <typename U> class FluidDescriptor>
ComputeFluxes<T, FluidDescriptor> *
ComputeFluxes<T, FluidDescriptor>::clone() const {
  return new ComputeFluxes<T, FluidDescriptor>(*this);
}

template <typename T, template <typename U> class FluidDescriptor>
void ComputeFluxes<T, FluidDescriptor>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;
  modified[1] = modif::staticVariables;
  modified[2] = modif::staticVariables;
  modified[3] = modif::staticVariables;
  modified[4] = modif::staticVariables;
  modified[5] = modif::staticVariables;
  modified[6] = modif::staticVariables;
  modified[7] = modif::nothing;
  modified[8] = modif::nothing;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Third order Runge-Kutta for the temporal derivation (used with the WENO3 for
// the particle field)
///////////////////////////////////////////////////////////////////////////////////////////////////////////

/* ******** RK3_Step1_functional3D ****************************************** */

template <typename T>
void RK3_Step1_functional3D<T>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {
  PLB_PRECONDITION(fields.size() == 3);
  TensorField3D<T, N> *phi_n = dynamic_cast<TensorField3D<T, N> *>(fields[0]);
  TensorField3D<T, N> *phi_n_adv =
      dynamic_cast<TensorField3D<T, N> *>(fields[1]);
  TensorField3D<T, N> *phi_1 = dynamic_cast<TensorField3D<T, N> *>(fields[2]);
  Dot3D offset_phi_n_adv = computeRelativeDisplacement(*phi_n, *phi_n_adv);
  Dot3D offset_phi_1 = computeRelativeDisplacement(*phi_n, *phi_1);
  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
        phi_1->get(iX + offset_phi_1.x, iY + offset_phi_1.y,
                   iZ + offset_phi_1.z) =
            phi_n->get(iX, iY, iZ) + phi_n_adv->get(iX + offset_phi_n_adv.x,
                                                    iY + offset_phi_n_adv.y,
                                                    iZ + offset_phi_n_adv.z);
      }
    }
  }
}

template <typename T>
RK3_Step1_functional3D<T> *RK3_Step1_functional3D<T>::clone() const {
  return new RK3_Step1_functional3D<T>(*this);
}

template <typename T>
void RK3_Step1_functional3D<T>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;         // phi_n
  modified[1] = modif::nothing;         // phi_n_adv
  modified[2] = modif::staticVariables; // phi_1
}

template <typename T>
BlockDomain::DomainT RK3_Step1_functional3D<T>::appliesTo() const {
  return BlockDomain::bulkAndEnvelope; // Everything is local, no communication
                                       // needed.
}

/* ******** RK3_Step2_functional3D ****************************************** */

template <typename T>
void RK3_Step2_functional3D<T>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {
  PLB_PRECONDITION(fields.size() == 4);
  TensorField3D<T, N> *phi_n = dynamic_cast<TensorField3D<T, N> *>(fields[0]);
  TensorField3D<T, N> *phi_1 = dynamic_cast<TensorField3D<T, N> *>(fields[1]);
  TensorField3D<T, N> *phi_1_adv =
      dynamic_cast<TensorField3D<T, N> *>(fields[2]);
  TensorField3D<T, N> *phi_2 = dynamic_cast<TensorField3D<T, N> *>(fields[3]);
  Dot3D offset_phi_1 = computeRelativeDisplacement(*phi_n, *phi_1);
  Dot3D offset_phi_1_adv = computeRelativeDisplacement(*phi_n, *phi_1_adv);
  Dot3D offset_phi_2 = computeRelativeDisplacement(*phi_n, *phi_2);
  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
        phi_2->get(iX + offset_phi_2.x, iY + offset_phi_2.y,
                   iZ + offset_phi_2.z) =
            3. / 4. * phi_n->get(iX, iY, iZ) +
            1. / 4. *
                phi_1->get(iX + offset_phi_1.x, iY + offset_phi_1.y,
                           iZ + offset_phi_1.z) +
            1. / 4. *
                phi_1_adv->get(iX + offset_phi_1_adv.x, iY + offset_phi_1_adv.y,
                               iZ + offset_phi_1_adv.z);
      }
    }
  }
}

template <typename T>
RK3_Step2_functional3D<T> *RK3_Step2_functional3D<T>::clone() const {
  return new RK3_Step2_functional3D<T>(*this);
}

template <typename T>
void RK3_Step2_functional3D<T>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;         // phi_n
  modified[1] = modif::nothing;         // phi_1
  modified[2] = modif::nothing;         // phi_1_adv
  modified[3] = modif::staticVariables; // phi_2
}

template <typename T>
BlockDomain::DomainT RK3_Step2_functional3D<T>::appliesTo() const {
  return BlockDomain::bulkAndEnvelope; // Everything is local, no communication
                                       // needed.
}

/* ******** RK3_Step3_functional3D ****************************************** */

template <typename T>
void RK3_Step3_functional3D<T>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {
  PLB_PRECONDITION(fields.size() == 4);
  TensorField3D<T, N> *phi_n = dynamic_cast<TensorField3D<T, N> *>(fields[0]);
  TensorField3D<T, N> *phi_2 = dynamic_cast<TensorField3D<T, N> *>(fields[1]);
  TensorField3D<T, N> *phi_2_adv =
      dynamic_cast<TensorField3D<T, N> *>(fields[2]);
  TensorField3D<T, N> *volfracField_RK =
      dynamic_cast<TensorField3D<T, N> *>(fields[3]);
  Dot3D offset_phi_2 = computeRelativeDisplacement(*phi_n, *phi_2);
  Dot3D offset_phi_2_adv = computeRelativeDisplacement(*phi_n, *phi_2_adv);
  Dot3D offset_volfracField_RK =
      computeRelativeDisplacement(*phi_n, *volfracField_RK);

  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
        volfracField_RK->get(iX + offset_volfracField_RK.x,
                             iY + offset_volfracField_RK.y,
                             iZ + offset_volfracField_RK.z) =
            1. / 3. * phi_n->get(iX, iY, iZ) +
            2. / 3. *
                phi_2->get(iX + offset_phi_2.x, iY + offset_phi_2.y,
                           iZ + offset_phi_2.z) +
            2. / 3. *
                phi_2_adv->get(iX + offset_phi_2_adv.x, iY + offset_phi_2_adv.y,
                               iZ + offset_phi_2_adv.z);
      }
    }
  }
}

template <typename T>
RK3_Step3_functional3D<T> *RK3_Step3_functional3D<T>::clone() const {
  return new RK3_Step3_functional3D<T>(*this);
}

template <typename T>
void RK3_Step3_functional3D<T>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;         // phi_n
  modified[1] = modif::nothing;         // phi_2
  modified[2] = modif::nothing;         // phi_2_adv
  modified[3] = modif::staticVariables; // volfracField_RK
}

template <typename T>
BlockDomain::DomainT RK3_Step3_functional3D<T>::appliesTo() const {
  return BlockDomain::bulkAndEnvelope; // Everything is local, no communication
                                       // needed.
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Regularization function for the volfracField_tot
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> Regu_VF_functional3D<T>::Regu_VF_functional3D() {}

template <typename T>
void Regu_VF_functional3D<T>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {
  PLB_PRECONDITION(fields.size() == 3);
  ScalarField3D<T> *volfracField_tot =
      dynamic_cast<ScalarField3D<T> *>(fields[0]);
  TensorField3D<T, N> *volfracField =
      dynamic_cast<TensorField3D<T, N> *>(fields[1]);
  TensorField3D<T, N> *volfracField_agg =
      dynamic_cast<TensorField3D<T, N> *>(fields[2]);
  Dot3D offset1 = computeRelativeDisplacement(*volfracField_tot, *volfracField);
  Dot3D offset2 =
      computeRelativeDisplacement(*volfracField_tot, *volfracField_agg);

  T vf_tmp, vf_tmp_agg;
  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
        vf_tmp = 0.;
        vf_tmp_agg = 0.;
        for (plint iD = 0; iD < N; ++iD) {
          vf_tmp += volfracField->get(iX + offset1.x, iY + offset1.y,
                                      iZ + offset1.z)[iD];
          vf_tmp_agg += volfracField_agg->get(iX + offset2.x, iY + offset2.y,
                                              iZ + offset2.z)[iD];
        }
        volfracField_tot->get(iX, iY, iZ) = vf_tmp + vf_tmp_agg;
      }
    }
  }
}

template <typename T>
Regu_VF_functional3D<T> *Regu_VF_functional3D<T>::clone() const {
  return new Regu_VF_functional3D<T>(*this);
}

template <typename T>
void Regu_VF_functional3D<T>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::staticVariables;
  modified[1] = modif::nothing;
}

template <typename T>
BlockDomain::DomainT Regu_VF_functional3D<T>::appliesTo() const {
  return BlockDomain::bulkAndEnvelope; // Everything is local, no communication
                                       // needed.
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Source terms
///////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, template <typename U> class FluidDescriptor>
BirthAndDeath<T, FluidDescriptor>::BirthAndDeath(
    const std::vector<std::vector<T>> &K_i_,
    const std::vector<std::vector<T>> &K_a_,
    const std::vector<std::vector<T>> &K_i_a_,
    const std::vector<std::vector<T>> &K_a_i_,
    const std::vector<std::vector<T>> &KTS_i_,
    const std::vector<std::vector<T>> &KTS_a_,
    const std::vector<std::vector<T>> &KTS_i_a_, T dx_, T dt_,
    const std::vector<T> &Dp_, const std::vector<T> &Dp_agg_,
    const std::vector<T> &m_,
    const std::vector<std::vector<std::vector<T>>> &coeff_,
    const std::vector<T> rhoPi_, const std::vector<T> rhoPa_)
    : K_i(K_i_), K_a(K_a_), K_i_a(K_i_a_), K_a_i(K_a_i_), KTS_i(KTS_i_),
      KTS_a(KTS_a_), KTS_i_a(KTS_i_a_), dx(dx_), dt(dt_), Dp(Dp_),
      Dp_agg(Dp_agg_), m(m_), coeff(coeff_), rhoPi(rhoPi_), rhoPa(rhoPa_) {}

template <typename T, template <typename U> class FluidDescriptor>
void BirthAndDeath<T, FluidDescriptor>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {

  PLB_PRECONDITION(fields.size() == 5);
  ScalarField3D<T> *dissipation = dynamic_cast<ScalarField3D<T> *>(fields[0]);
  TensorField3D<T, N> *volfracfield =
      dynamic_cast<TensorField3D<T, N> *>(fields[1]);
  TensorField3D<T, N> *Q = dynamic_cast<TensorField3D<T, N> *>(fields[2]);
  TensorField3D<T, N> *volfracfield_agg =
      dynamic_cast<TensorField3D<T, N> *>(fields[3]);
  TensorField3D<T, N> *Q_agg = dynamic_cast<TensorField3D<T, N> *>(fields[4]);

  Dot3D offset1 = computeRelativeDisplacement(*dissipation, *volfracfield);
  Dot3D offset2 = computeRelativeDisplacement(*dissipation, *Q);
  Dot3D offset3 = computeRelativeDisplacement(*dissipation, *volfracfield_agg);
  Dot3D offset4 = computeRelativeDisplacement(*dissipation, *Q_agg);

  Array<T, N> Bi, Di;
  Array<T, N> Bi_agg, Di_agg, Phi_agg, Ni_agg, Q_tmp_agg, vf_tmp_agg;
  for (plint i = 0; i < N; ++i) {
    Bi[i] = 0.;
    Di[i] = 0.;
    Bi_agg[i] = 0.;
    Di_agg[i] = 0.;
  }
  Array<T, N> Phi, Ni, Q_tmp, vf_tmp;
  T Bi_i, Ba_i, Ba_a, Bi_a, Di_i, Da_i, Di_a, Da_a;
  T epsilon;
  T vf, vf_agg;
  T sc = (pow(dx, 2)/pow(dt, 3));

  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {

        vf_tmp =
            volfracfield->get(iX + offset1.x, iY + offset1.y, iZ + offset1.z);
        vf_tmp_agg = volfracfield_agg->get(iX + offset3.x, iY + offset3.y,
                                           iZ + offset3.z);
        for (plint iD = 0; iD < N; ++iD) {
          vf = (util::greaterThan(vf_tmp[iD], (T)0)
                    ? vf_tmp[iD]
                    : (util::lessThan(vf_tmp[iD], (T)0) ? 0 : 0));
          vf_agg = (util::greaterThan(vf_tmp_agg[iD], (T)0)
                        ? vf_tmp_agg[iD]
                        : (util::lessThan(vf_tmp_agg[iD], (T)0) ? 0 : 0));
          Phi[iD] = vf;
          Phi_agg[iD] = vf_agg;
          Ni[iD] = Phi[iD] * rhoPi[iD] / m[iD];
          Ni_agg[iD] = Phi_agg[iD] * rhoPa[iD] / m[iD];
        }

        epsilon = sc*dissipation->get(iX,iY,iZ);
        // epsilon = 0.;

        // Computation of the Birth term
        for (plint i = 0; i < N; ++i) {
          Bi_i = 0.;
          Ba_i = 0.;
          Ba_a = 0.;
          Bi_a = 0.;
          Di_i = 0.;
          Da_i = 0.;
          Di_a = 0.;
          Da_a = 0.;
          for (plint j = 0; j < N; ++j) {
            for (plint k = 0; k <= j; ++k) {
              Bi_i =
                  Bi_i + coeff[i][j][k] *
                             (K_i[k][j] + pow(epsilon, (T)0.5) * KTS_i[k][j]) *
                             Ni[k] * Ni[j];
              Ba_i = Ba_i +
                     coeff[i][j][k] *
                         (K_a_i[k][j] + pow(epsilon, (T)0.5) * KTS_i_a[k][j]) *
                         Ni_agg[k] * Ni[j];
              Ba_a =
                  Ba_a + coeff[i][j][k] *
                             (K_a[k][j] + pow(epsilon, (T)0.5) * KTS_a[k][j]) *
                             Ni_agg[k] * Ni_agg[j];
              Bi_a = Bi_a +
                     coeff[i][j][k] *
                         (K_i_a[k][j] + pow(epsilon, (T)0.5) * KTS_i_a[k][j]) *
                         Ni[k] * Ni_agg[j];
            }
            Di_i = Di_i + (K_i[i][j] + pow(epsilon, (T)0.5) * KTS_i[i][j]) *
                              Ni[i] * Ni[j];
            Da_i = Da_i + (K_a_i[i][j] + pow(epsilon, (T)0.5) * KTS_i_a[i][j]) *
                              Ni_agg[i] * Ni[j];
            Di_a = Di_a + (K_i_a[i][j] + pow(epsilon, (T)0.5) * KTS_i_a[i][j]) *
                              Ni[i] * Ni_agg[j];
            Da_a = Da_a + (K_a[i][j] + pow(epsilon, (T)0.5) * KTS_a[i][j]) *
                              Ni_agg[i] * Ni_agg[j];
          }
          Bi[i] = 0.;
          Di[i] = Di_i + Di_a;
          Bi_agg[i] = Bi_i + Ba_a + Bi_a + Ba_i;
          if (i != N - 1) {
            Di_agg[i] = Da_a + Da_i;
          } else {
            Di_agg[i] = 0.;
          }
        }

        // Computation of the total source term
        for (plint iD = 0; iD < N; ++iD) {
          Q_tmp[iD] = (m[iD] / rhoPi[iD]) * (Bi[iD] - Di[iD]) * (dt);
          Q_tmp_agg[iD] =
              (m[iD] / rhoPa[iD]) * (Bi_agg[iD] - Di_agg[iD]) * (dt);
        }
        Q->get(iX + offset2.x, iY + offset2.y, iZ + offset2.z) = Q_tmp;
        Q_agg->get(iX + offset4.x, iY + offset4.y, iZ + offset4.z) = Q_tmp_agg;
        Phi.resetToZero();
        Ni.resetToZero();
        Bi.resetToZero();
        Di.resetToZero();
        Phi_agg.resetToZero();
        Ni_agg.resetToZero();
        Bi_agg.resetToZero();
        Di_agg.resetToZero();
      }
    }
  }
}

template <typename T, template <typename U> class FluidDescriptor>
BirthAndDeath<T, FluidDescriptor> *
BirthAndDeath<T, FluidDescriptor>::clone() const {
  return new BirthAndDeath<T, FluidDescriptor>(*this);
}

template <typename T, template <typename U> class FluidDescriptor>
void BirthAndDeath<T, FluidDescriptor>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;
  modified[1] = modif::nothing;
  modified[2] = modif::staticVariables;
  modified[3] = modif::nothing;
  modified[4] = modif::staticVariables;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dissipation Rate
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, template <typename U> class FluidDescriptor>
ComputeDissipationRate<T, FluidDescriptor>::ComputeDissipationRate(T cSmago_)
    : cSmago(cSmago_) {}

template <typename T, template <typename U> class FluidDescriptor>
void ComputeDissipationRate<T, FluidDescriptor>::processGenericBlocks(
    Box3D domain, std::vector<AtomicBlock3D *> fields) {

  PLB_PRECONDITION(fields.size() == 2);
  BlockLattice3D<T, FluidDescriptor> *fluid =
      dynamic_cast<BlockLattice3D<T, FluidDescriptor> *>(fields[0]);
  ScalarField3D<T> *dissipation = dynamic_cast<ScalarField3D<T> *>(fields[1]);

  Dot3D offset1 = computeRelativeDisplacement(*fluid, *dissipation);

  T dissRate;

  for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
    for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
      for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
        Cell<T, FluidDescriptor> const &cell = fluid->get(iX, iY, iZ);
        Array<T, 6> S, P;
        cell.computePiNeq(P);
        T omega = cell.getDynamics().getOmega();
        for (int i = 0; i < 6; ++i) {
          S[i] = -(T)0.5 * omega * FluidDescriptor<T>::invCs2 * P[i];
        }

        T SNormSqr = SymmetricTensor<T, FluidDescriptor>::tensorNormSqr(S);
        dissRate = cSmago * cSmago * SNormSqr * std::sqrt(SNormSqr);
        dissipation->get(iX + offset1.x, iY + offset1.y, iZ + offset1.z) =
            dissRate;
      }
    }
  }
}

template <typename T, template <typename U> class FluidDescriptor>
ComputeDissipationRate<T, FluidDescriptor> *
ComputeDissipationRate<T, FluidDescriptor>::clone() const {
  return new ComputeDissipationRate<T, FluidDescriptor>(*this);
}

template <typename T, template <typename U> class FluidDescriptor>
void ComputeDissipationRate<T, FluidDescriptor>::getTypeOfModification(
    std::vector<modif::ModifT> &modified) const {
  modified[0] = modif::nothing;
  modified[1] = modif::staticVariables;
}

} // namespace plb
#endif
