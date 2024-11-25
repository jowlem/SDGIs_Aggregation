#include "palabos3D.h"
#include "palabos3D.hh"
#include "headers.hh"
#include "headers.h"

#include <cstdlib>
#include <iostream>
#include <random>
#include <functional>
#include <fstream>
#include <filesystem>
#include <climits>

using namespace plb;
using namespace std;

typedef double T;
#define N 32 // Number of bins

// Choice of the descriptor to define the dimension and the velocity space
#define NSDESCRIPTOR descriptors::ForcedD3Q19Descriptor

// Choice of the Lattice Boltzmann Dynamics
#define NSDYNAMICS GuoExternalForceConsistentSmagorinskyCompleteRegularizedBGKdynamics

////////////////////////////////////////////////////
/// Initialization of the volume fraction field. ///
////////////////////////////////////////////////////

template <typename T, template <typename NSU> class nsDescriptor>
struct IniVolFracProcessor3D : public BoxProcessingFunctional3D_T<T, N> {
  IniVolFracProcessor3D(ConvertToKmMin<T, nsDescriptor> parameters_,
                        std::vector<T> w_, T TotalVolFrac_)
      : parameters(parameters_), w(w_), TotalVolFrac(TotalVolFrac_) {}
  virtual void process(Box3D domain, TensorField3D<T, N> &volfracField) {
    Dot3D absoluteOffset = volfracField.getLocation();

    T nz = parameters.getNz();
    T up = parameters.getUp();
    T low = parameters.getLow();

    for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
      for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
        for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
          plint absoluteZ = absoluteOffset.z + iZ;
          Array<T, N> VolFrac;
          for (plint iD = 0; iD < N; ++iD) {
            std::random_device rd;
            std::mt19937 mt(rd());
            std::normal_distribution<> dist(1, 0.01);

            // Inject a slight noise in the distribution to trigger SDGIs
            T rand_val = dist(mt);

            // Uniformly distributed
            // T rand_val = 1.;

            if (absoluteZ <
                    (nz - 1) - util::roundToInt(((up / (up + low)) * nz)) ||
                absoluteZ >= (nz - 6))
              VolFrac[iD] = 0.0;
            else
              VolFrac[iD] = TotalVolFrac * w[iD] * rand_val;
          }
          volfracField.get(iX, iY, iZ) = VolFrac;
        }
      }
    }
  }
  virtual IniVolFracProcessor3D<T, nsDescriptor> *clone() const {
    return new IniVolFracProcessor3D<T, nsDescriptor>(*this);
  }
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const {
    modified[0] = modif::staticVariables;
  }
  virtual BlockDomain::DomainT appliesTo() const {
    return BlockDomain::bulkAndEnvelope;
  }

private:
  ConvertToKmMin<T, nsDescriptor> parameters;
  std::vector<T> w;
  T TotalVolFrac;
};

////////////////////////////////////////////
/// Initialization of the density field. ///
////////////////////////////////////////////

template <typename T, template <typename NSU> class nsDescriptor>
struct IniDensityProcessor3D : public BoxProcessingFunctional3D_S<T> {
  IniDensityProcessor3D(ConvertToKmMin<T, nsDescriptor> parameters_)
      : parameters(parameters_) {}
  virtual void process(Box3D domain, ScalarField3D<T> &densityField) {
    Dot3D absoluteOffset = densityField.getLocation();

    T nz = parameters.getNz();
    T up = parameters.getUp();
    T low = parameters.getLow();

    for (plint iX = domain.x0; iX <= domain.x1; ++iX) {
      for (plint iY = domain.y0; iY <= domain.y1; ++iY) {
        for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ) {
          plint absoluteZ = absoluteOffset.z + iZ;
          T dens;

          if (absoluteZ < (nz - 1) - util::roundToInt(((up / (up + low)) * nz)))
            dens = parameters.getRhoLow();
          else
            dens = parameters.getRhoUp();

          densityField.get(iX, iY, iZ) = dens;
        }
      }
    }
  }
  virtual IniDensityProcessor3D<T, nsDescriptor> *clone() const {
    return new IniDensityProcessor3D<T, nsDescriptor>(*this);
  }
  virtual void
  getTypeOfModification(std::vector<modif::ModifT> &modified) const {
    modified[0] = modif::staticVariables;
  }
  virtual BlockDomain::DomainT appliesTo() const {
    return BlockDomain::bulkAndEnvelope;
  }

private:
  ConvertToKmMin<T, nsDescriptor> parameters;
};

//////////////////////////////////////////////////////////////////
/// A function which performs a global setup of the simulation ///
//////////////////////////////////////////////////////////////////

void ExpSetup(MultiBlockLattice3D<T, NSDESCRIPTOR> &nsLattice,
              std::vector<MultiBlock3D *> fields, std::vector<T> weights,
              T TotalVolFrac, MultiScalarField3D<T> &densityField,
              MultiScalarField3D<T> &tmp,
              ConvertToKmMin<T, NSDESCRIPTOR> &parameters) {
  PLB_PRECONDITION(atomicBlocks.size() == 2);
  MultiScalarField3D<T> *volfracField_tot =
      dynamic_cast<MultiScalarField3D<T> *>(fields[0]);
  MultiTensorField3D<T, N> *volfracField =
      dynamic_cast<MultiTensorField3D<T, N> *>(fields[1]);

  // Initialize a particle volume fraction field for each bin
  applyProcessingFunctional(new IniVolFracProcessor3D<T, NSDESCRIPTOR>(
                                parameters, weights, TotalVolFrac),
                            volfracField->getBoundingBox(), *volfracField);

  // Initialize the density field
  applyProcessingFunctional(
      new IniDensityProcessor3D<T, NSDESCRIPTOR>(parameters),
      densityField.getBoundingBox(), densityField);

  // A function which calculates the total volume fraction field
  applyProcessingFunctional(new Regu_VF_functional3D<T>(),
                            volfracField_tot->getBoundingBox(), fields);

  // Initialize the fluid field (Lattice Boltzmann Method)
  initializeAtEquilibrium(nsLattice, nsLattice.getBoundingBox(), (T)1.,
                          Array<T, 3>((T)0., (T)0., (T)0.));
  nsLattice.initialize();
}

/////////////////////////////////////////////////////////////////////////////
/// Compute the aggregation kernel associated with Differential settling. ///
/////////////////////////////////////////////////////////////////////////////
void ComputeK(std::vector<std::vector<T>> &K_i,
              std::vector<std::vector<T>> &K_a,
              std::vector<std::vector<T>> &K_i_a,
              std::vector<std::vector<T>> &K_a_i,
              ConvertToKmMin<T, NSDESCRIPTOR> const &parameters,
              const std::vector<T> &Dp, const std::vector<T> &rhoPi,
              const std::vector<T> &rhoPa, const std::vector<T> &Dp_agg) {
  T c1 = 1.;
  T Stcr = 1.3;
  T rhoLow = 1.2353;
  T q = 1.6;
  T g = 9.81;
  T mu = 1.91e-5;
  T vs_i, St_i;
  T vs_a, St_a, St_i_a, St_a_i;

  std::vector<T> Vs_i;
  std::vector<T> Vs_a;

  for (int i = 0; i < N; i++) {
    vs_i = (0.5 * Dp[i] * Dp[i] * g * (rhoPi[i] - rhoLow)) / (9 * mu);
    Vs_i.push_back(vs_i);
    vs_a = (0.5 * Dp_agg[i] * Dp_agg[i] * g * (rhoPa[i] - rhoLow)) / (9 * mu);
    Vs_a.push_back(vs_a);
  }

  std::vector<std::vector<T>> alpha_i(N, std::vector<T>(N, 0));
  std::vector<std::vector<T>> alpha_a(N, std::vector<T>(N, 0));
  std::vector<std::vector<T>> alpha_i_a(N, std::vector<T>(N, 0));
  std::vector<std::vector<T>> alpha_a_i(N, std::vector<T>(N, 0));
  std::vector<std::vector<T>> beta_DS_i(N, std::vector<T>(N, 0));
  std::vector<std::vector<T>> beta_DS_a(N, std::vector<T>(N, 0));
  std::vector<std::vector<T>> beta_DS_i_a(N, std::vector<T>(N, 0));
  std::vector<std::vector<T>> beta_DS_a_i(N, std::vector<T>(N, 0));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      St_i = ((8. * rhoPi[i] * abs(Vs_i[i] - Vs_i[j])) / (9. * mu)) *
             ((Dp[i] * Dp[j]) / (Dp[i] + Dp[j]));
      St_a = ((8. * rhoPa[i] * abs(Vs_a[i] - Vs_a[j])) / (9. * mu)) *
             ((Dp_agg[i] * Dp_agg[j]) / (Dp_agg[i] + Dp_agg[j]));
      St_i_a = ((8. * ((rhoPi[i] + rhoPa[j]) / 2.) * abs(Vs_i[i] - Vs_a[j])) /
                (9. * mu)) *
               ((Dp[i] * Dp_agg[j]) / (Dp[i] + Dp_agg[j]));
      St_a_i = ((8. * ((rhoPa[i] + rhoPi[j]) / 2.) * abs(Vs_a[i] - Vs_i[j])) /
                (9. * mu)) *
               ((Dp_agg[i] * Dp[j]) / (Dp_agg[i] + Dp[j]));

      alpha_i[i][j] = c1 * (1 / (pow((St_i / Stcr), q) + 1));
      alpha_a[i][j] = c1 * (1 / (pow((St_a / Stcr), q) + 1));
      alpha_i_a[i][j] = c1 * (1 / (pow((St_i_a / Stcr), q) + 1));
      alpha_a_i[i][j] = c1 * (1 / (pow((St_a_i / Stcr), q) + 1));

      beta_DS_i[i][j] =
          ((M_PI / (T)4) * pow((Dp[i] + Dp[j]), 2) * abs(Vs_i[i] - Vs_i[j])) *
          (60 * (1e-9));
      beta_DS_a[i][j] = ((M_PI / (T)4) * pow((Dp_agg[i] + Dp_agg[j]), 2) *
                         abs(Vs_a[i] - Vs_a[j])) *
                        (60 * (1e-9));
      beta_DS_i_a[i][j] = ((M_PI / (T)4) * pow((Dp[i] + Dp_agg[j]), 2) *
                           abs(Vs_i[i] - Vs_a[j])) *
                          (60 * (1e-9));
      beta_DS_a_i[i][j] = ((M_PI / (T)4) * pow((Dp_agg[i] + Dp[j]), 2) *
                           abs(Vs_a[i] - Vs_i[j])) *
                          (60 * (1e-9));

      K_i[i][j] = alpha_i[i][j] * beta_DS_i[i][j];
      K_a[i][j] = alpha_a[i][j] * beta_DS_a[i][j];
      K_i_a[i][j] = alpha_i_a[i][j] * beta_DS_i_a[i][j];
      K_a_i[i][j] = alpha_a_i[i][j] * beta_DS_a_i[i][j];

      // For a fixed kernel

      // K_i[i][j] = 1.e-20;
      // K_a[i][j] = 1.e-20;
      // K_i_a[i][j] = 1.e-20;
      // K_a_i[i][j] = 1.e-20;
    }
  }
}

///////////////////////////////////////////////////////////////////////
/// Compute the aggregation kernel associated with Turbulent shear. ///
///////////////////////////////////////////////////////////////////////
void ComputeK_TS(std::vector<std::vector<T>> &KTS_i,
                 std::vector<std::vector<T>> &KTS_a,
                 std::vector<std::vector<T>> &KTS_i_a,
                 ConvertToKmMin<T, NSDESCRIPTOR> const &parameters,
                 const std::vector<T> &Dp) {
  T c1 = 1.;
  T Stcr = 1.3;
  T rhoLow = 1.2353;
  T rhoPi = 2500.;
  T rhoPa = 1000.;
  T rhoP_avg = (rhoPi + rhoPa) / 2;
  T q = 1.6;
  T g = 9.81;
  T nuPh = 15.1e-6;
  T mu = 1.91e-5;
  T vs_i, St_i;
  T vs_a, St_a, St_i_a;

  std::vector<T> Vs_i;
  std::vector<T> Vs_a;

  for (int i = 0; i < N; i++) {
    vs_i = (0.5 * Dp[i] * Dp[i] * g * (rhoPi - rhoLow)) / (9 * mu);
    Vs_i.push_back(vs_i);
    vs_a = (0.5 * Dp[i] * Dp[i] * g * (rhoPa - rhoLow)) / (9 * mu);
    Vs_a.push_back(vs_a);
  }

  std::vector<std::vector<T>> alpha_i(N, std::vector<T>(N, 0.));
  std::vector<std::vector<T>> alpha_a(N, std::vector<T>(N, 0.));
  std::vector<std::vector<T>> alpha_i_a(N, std::vector<T>(N, 0.));
  std::vector<std::vector<T>> beta_TS(N, std::vector<T>(N, 0.));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      St_i = ((8 * rhoPi * abs(Vs_i[i] - Vs_i[j])) / (9 * mu)) *
             ((Dp[i] * Dp[j]) / (Dp[i] + Dp[j]));
      St_a = ((8 * rhoPa * abs(Vs_a[i] - Vs_a[j])) / (9 * mu)) *
             ((Dp[i] * Dp[j]) / (Dp[i] + Dp[j]));
      St_i_a = ((8 * rhoP_avg * abs(Vs_i[i] - Vs_a[j])) / (9 * mu)) *
               ((Dp[i] * Dp[j]) / (Dp[i] + Dp[j]));

      alpha_i[i][j] = c1 * (1 / (pow((St_i / Stcr), q) + 1));
      alpha_a[i][j] = c1 * (1 / (pow((St_a / Stcr), q) + 1));
      alpha_i_a[i][j] = c1 * (1 / (pow((St_i_a / Stcr), q) + 1));

      beta_TS[i][j] =
          (pow(M_PI / ((T)15 * nuPh), (T)0.5) * pow(Dp[i] + Dp[j], 3)) *
          (60 * (1e-9));

      KTS_i[i][j] = alpha_i[i][j] * beta_TS[i][j];
      KTS_a[i][j] = alpha_a[i][j] * beta_TS[i][j];
      KTS_i_a[i][j] = alpha_i_a[i][j] * beta_TS[i][j];

      // For a fixed kernel
      // KTS_i[i][j] = 0.;
      // KTS_a[i][j] = 0.;
      // KTS_i_a[i][j] = 0.;
    }
  }
}

//////////////////////////////////////////////////////////////////
/// Calculate the total mass of particles in the whole domain. ///
//////////////////////////////////////////////////////////////////
T ComputeTotalMass(MultiTensorField3D<T, N> &volfracField,
                   MultiTensorField3D<T, N> &volfracField_agg,
                   ConvertToKmMin<T, NSDESCRIPTOR> const &parameters) {
  Array<T, N> m, m_agg;
  T m_tot = 0.;
  std::vector<T> rhoPi = parameters.getRhoPi();
  std::vector<T> rhoPa = parameters.getRhoPa();
  for (plint iD = 0; iD < N; ++iD) {
    m_tot += pow(parameters.getDeltaX(), 3) *
             (computeSum(*extractComponent(volfracField, iD)) * rhoPi[iD] +
              computeSum(*extractComponent(volfracField_agg, iD)) * rhoPa[iD]);
  }
  return m_tot;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the coefficient matrix involving the mass of each bin in Eq. 11. ///
////////////////////////////////////////////////////////////////////////////////
void ComputeCoeffs(const std::vector<T> &m,
                   std::vector<std::vector<std::vector<T>>> &coeff) {
  T m_t;
  T coeff1, coeff2;
  for (plint i = 0; i < N; i++) {
    for (plint j = 0; j < N; j++) {
      for (plint k = 0; k <= j; k++) {
        coeff1 = 0.;
        coeff2 = 0.;
        m_t = m[k] + m[j];
        if (i != N - 1) {
          if (m[i] <= m_t && m_t < m[i + 1])
            coeff1 = ((T)1. - (T)(util::kronDelta(j, k) / 2.)) *
                     ((m[i + 1] - m_t) / (m[i + 1] - m[i]));
        }
        if (i != 0) {
          if (m[i - 1] <= m_t && m_t < m[i])
            coeff2 = ((T)1. - (T)(util::kronDelta(j, k) / 2.)) *
                     ((m_t - m[i - 1]) / (m[i] - m[i - 1]));
        }
        coeff[i][j][k] = coeff1 + coeff2;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////
/// writeVTK (for Paraview) and writeGif (images) are functions that save data
/// /// for visualization. ///
//////////////////////////////////////////////////////////////////////////////////
void writeVTK(MultiBlockLattice3D<T, NSDESCRIPTOR> &nsLattice,
              std::vector<MultiBlock3D *> fields,
              ConvertToKmMin<T, NSDESCRIPTOR> const &parameters, plint iter) {
  PLB_PRECONDITION(atomicBlocks.size() == 7);
  MultiScalarField3D<T> *volfracField_tot =
      dynamic_cast<MultiScalarField3D<T> *>(fields[0]);
  MultiTensorField3D<T, N> *volfracField =
      dynamic_cast<MultiTensorField3D<T, N> *>(fields[1]);
  MultiTensorField3D<T, N> *Q =
      dynamic_cast<MultiTensorField3D<T, N> *>(fields[2]);
  MultiTensorField3D<T, N> *v_sedimentation =
      dynamic_cast<MultiTensorField3D<T, N> *>(fields[3]);
  MultiScalarField3D<T> *dissipation =
      dynamic_cast<MultiScalarField3D<T> *>(fields[4]);
  MultiTensorField3D<T, N> *volfracField_agg =
      dynamic_cast<MultiTensorField3D<T, N> *>(fields[5]);
  MultiTensorField3D<T, N> *Q_agg =
      dynamic_cast<MultiTensorField3D<T, N> *>(fields[6]);

  T dx = parameters.getDeltaX();
  T dt = parameters.getDeltaT();
  std::vector<T> rhoPi = parameters.getRhoPi();
  VtkImageOutput3D<T> vtkOut(createFileName("vtk", iter, 6), dx);
  vtkOut.writeData<3, float>(*computeVelocity(nsLattice), "NSvelocity",
                             dx / dt);
  vtkOut.writeData<N, float>(*volfracField, "VolFrac", (T)1.);
  vtkOut.writeData<float>(*volfracField_tot, "VolFractot", (T)1.);
  vtkOut.writeData<N, float>(*v_sedimentation, "v_sed", (T)1.);
  vtkOut.writeData<N, float>(*Q, "Q", (T)1.);
  vtkOut.writeData<N, float>(*volfracField, "Mass",
                             (T)pow(parameters.getDeltaX(), 3) * rhoPi[0]);
  vtkOut.writeData<float>(*dissipation, "Dissipation rate",
                          (T)(1000 * dx * 1000 * dx) /
                              (60 * dt * 60 * dt * 60 * dt));
  vtkOut.writeData<N, float>(*volfracField_agg, "VolFrac_agg", (T)1.);
  vtkOut.writeData<N, float>(*Q_agg, "Q_agg", (T)1.);
}

void writeGif(MultiBlockLattice3D<T, NSDESCRIPTOR> &nsLattice,
              MultiScalarField3D<T> &volfracField_tot,
              MultiScalarField3D<T> &densityField, int iT) {
  const plint imSize = 600;
  const plint nx = nsLattice.getNx();
  const plint ny = nsLattice.getNy();
  const plint nz = nsLattice.getNz();
  Box3D slice((nx - 1) / 2, (nx - 1) / 2, 0, ny - 1, 0, nz - 1);
  ImageWriter<T> imageWriter("leeloo.map");
  imageWriter.writeScaledGif(createFileName("u", iT, 6),
                             *computeVelocityNorm(nsLattice, slice), imSize,
                             imSize);
  imageWriter.writeScaledGif(createFileName("VolFrac", iT, 6),
                             *extractSubDomain(volfracField_tot, slice), imSize,
                             imSize);

  imageWriter.writeScaledGif(createFileName("Density", iT, 6),
                             *extractSubDomain(densityField, slice), imSize,
                             imSize);

  imageWriter.writeScaledGif(
      createFileName("Vorticity", iT, 6),
      *computeNorm(*computeVorticity(*computeVelocity(nsLattice)), slice),
      imSize, imSize);
}

int main(int argc, char *argv[]) {
  plbInit(&argc, &argv);
  global::timer("simTime").start();

  // input gaussian grainsize distribution for 30 bins
  ifstream file("input_phi_dist30_gau_6.txt");

  // input uniform grainsize distribution for 30 bins
  // ifstream file("input_phi_dist30_all.txt");

  // input monodisperse grainsize distribution for 8 bins
  // + 2 trash bins for mass conservation
  // ifstream file("Validation_mono_8_bins.txt");

  // input polydisperse grainsize distribution for 8 bins
  // ifstream file("Validation_poly_8_bins.txt");

  // input polydisperse grainsize distribution for 30 bins
  // ifstream file("Validation_poly_30_bins.txt");

  if (!file) {
    cerr << "cannot read the file input.txt :" << strerror(errno) << endl;
    return -1;
  }

  std::vector<T> Dp;
  std::vector<T> Dp_agg(N, 0.);
  std::vector<T> weights;
  T n1, n2;
  T scale = 1e12;

  while (file >> n1 >> n2) {
    Dp.insert(Dp.begin(), (1.e-3) * pow(2., -n1));
    weights.insert(weights.begin(), n2 / 100);
  }

  // Physical units in (m, s)
  const T up = 135;
  const T low = 250;
  const T rhoUp = 1.225;
  const T rhoLow = 1.2353;
  const T lx = 75;
  const T ly = 150;
  const T lz = 385;
  const T uCar = 4.1667;
  const T Di = 0.;
  std::vector<T> rhoPi(N, 0.);
  std::vector<T> rhoPa(N, 0.);
  for (int i = 0; i < N; ++i) {
    rhoPi[i] = 2500.; // Define here the individual particles density
    rhoPa[i] = 1000.; // Define here the aggregates density
  }
  T g = 9.81;
  T mu = 18.3e-6;
  const T Ri = 1.0;
  const T Gr = 1.125e16;
  const T uMax = 0.1;

  // Physical units in (km, min)

  const T up_k = up / 1000.;
  const T low_k = low / 1000.;
  const T rhoUp_k = rhoUp * (T)1e9;
  const T rhoLow_k = rhoLow * (T)1e9;
  const T lx_k = lx / 1000.;
  const T ly_k = ly / 1000.;
  const T lz_k = lz / 1000.;
  const T uCar_k = uCar * (T)0.06;
  const T Di_k = Di * (T)6e-5;
  std::vector<T> rhoPi_k = rhoPi;
  std::vector<T> rhoPa_k = rhoPa;
  transform(rhoPi_k.begin(), rhoPi_k.end(), rhoPi_k.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, 1e9));
  transform(rhoPa_k.begin(), rhoPa_k.end(), rhoPa_k.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, 1e9));
  T g_k = g * (T)3.6;
  T mu_k = mu * (T)6e4;
  std::vector<T> rhoP_k;
  const T TotalVolFrac = (2e-6);
  std::vector<T> Dip_k;
  std::vector<T> Dp_k = Dp;
  transform(Dp_k.begin(), Dp_k.end(), Dp_k.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, 1e-3));

  const plint resolution = 500; // The numerical resolution

  global::directories().setOutputDir("./tmp/");

  ConvertToKmMin<T, NSDESCRIPTOR> parameters(
      up_k, low_k, rhoUp_k, rhoLow_k, lx_k, ly_k, lz_k, uCar_k, Di_k, Dp_k,
      rhoPi_k, rhoPa_k, g_k, mu_k, Ri, Gr, uMax, resolution);

  std::vector<T> m1;
  for (int i = 0; i < N; ++i) {
    m1.push_back((rhoPi[i] * M_PI / (T)6) * pow(Dp[i], 3));
  }

  const std::vector<T> m = m1;

  for (int i = 0; i < N; ++i) {
    Dp_agg[i] = pow((6. * m[i]) / (rhoPa[i] * M_PI), 1. / 3.);
  }

  std::vector<T> Dp_agg_k = Dp_agg;
  transform(Dp_agg_k.begin(), Dp_agg_k.end(), Dp_agg_k.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, 1e-3));

  for (int i = 0; i < N; ++i) {
    pcout << Dp[i] / (1e-6) << "  " << Dp_agg[i] / (1e-6) << "  " << rhoPa[i]
          << endl;
  }

  std::vector<std::vector<T>> K_i(N, std::vector<T>(N, 0)),
      K_a(N, std::vector<T>(N, 0)), K_i_a(N, std::vector<T>(N, 0)),
      K_a_i(N, std::vector<T>(N, 0));
  std::vector<std::vector<T>> KTS_i(N, std::vector<T>(N, 0)),
      KTS_a(N, std::vector<T>(N, 0)), KTS_i_a(N, std::vector<T>(N, 0));
  ComputeK(K_i, K_a, K_i_a, K_a_i, parameters, Dp, rhoPi, rhoPa, Dp_agg);
  ComputeK_TS(KTS_i, KTS_a, KTS_i_a, parameters, Dp);

  std::vector<std::vector<std::vector<T>>> coeff(
      N, std::vector<std::vector<T>>(N, std::vector<T>(N, 0.)));
  ComputeCoeffs(m, coeff);
  rhoPi = parameters.getRhoPi();
  rhoPa = parameters.getRhoPa();
  g = parameters.getG();
  T g_lb = g_k * parameters.getLatticeGravity();
  T eps = 1e-6;

  writeLogFile(parameters, "palabos.log");

  plint nx = parameters.getNx();
  plint ny = parameters.getNy();
  plint nz = parameters.getNz();

  T nsOmega = parameters.getSolventOmega();
  T convers = parameters.getDeltaT() / parameters.getDeltaX();
  mu = parameters.getMu();
  T cSmago = 0.14;

  Array<T, N> alpha_x, alpha_y, alpha_z;

  plint envelopeWidth = 2;
  SparseBlockStructure3D blockStructure(
      createRegularDistribution3D(nx, ny, nz));

  Dynamics<T, NSDESCRIPTOR> *nsdynamics =
      new NSDYNAMICS<T, NSDESCRIPTOR>(nsOmega, cSmago);
  MultiBlockLattice3D<T, NSDESCRIPTOR> nsLattice(nx, ny, nz,
                                                 nsdynamics->clone());
  defineDynamics(nsLattice, nsLattice.getBoundingBox(), nsdynamics->clone());
  delete nsdynamics;
  nsdynamics = 0;

  MultiTensorField3D<T, 3> velocity(
      MultiBlockManagement3D(blockStructure,
                             defaultMultiBlockPolicy3D().getThreadAttribution(),
                             envelopeWidth),
      defaultMultiBlockPolicy3D().getBlockCommunicator(),
      defaultMultiBlockPolicy3D().getCombinedStatistics(),
      defaultMultiBlockPolicy3D().getMultiTensorAccess<T, 3>());

  MultiTensorField3D<T, N> volfracField(
      MultiBlockManagement3D(blockStructure,
                             defaultMultiBlockPolicy3D().getThreadAttribution(),
                             envelopeWidth),
      defaultMultiBlockPolicy3D().getBlockCommunicator(),
      defaultMultiBlockPolicy3D().getCombinedStatistics(),
      defaultMultiBlockPolicy3D().getMultiTensorAccess<T, N>());

  MultiScalarField3D<T> volfracField_tot(
      MultiBlockManagement3D(blockStructure,
                             defaultMultiBlockPolicy3D().getThreadAttribution(),
                             envelopeWidth),
      defaultMultiBlockPolicy3D().getBlockCommunicator(),
      defaultMultiBlockPolicy3D().getCombinedStatistics(),
      defaultMultiBlockPolicy3D().getMultiScalarAccess<T>(), T());

  MultiScalarField3D<T> densityField(volfracField_tot);
  MultiScalarField3D<T> dissipation(volfracField_tot);
  MultiScalarField3D<T> tmp(volfracField_tot);

  MultiScalarField3D<T> D_t(volfracField_tot), D_tp1(volfracField_tot),
      Q_d(volfracField_tot);

  MultiTensorField3D<T, N> v_sedimentation(volfracField),
      volfracField_RK(volfracField), Q(volfracField), phi_1(volfracField),
      phi_2(volfracField), phi_n_adv(volfracField), phi_1_adv(volfracField),
      phi_2_adv(volfracField), T_t(volfracField);

  MultiScalarField3D<T> velX(volfracField), velY(volfracField),
      velZ(volfracField);
  MultiTensorField3D<T, N> velX_t(volfracField), velY_t(volfracField),
      velZ_t(volfracField);
  MultiTensorField3D<T, N> Flux_px(volfracField), Flux_py(volfracField),
      Flux_pz(volfracField), Flux_nx(volfracField), Flux_ny(volfracField),
      Flux_nz(volfracField);

  MultiTensorField3D<T, N> volfracField_agg(volfracField),
      v_sedimentation_agg(volfracField), Q_agg(volfracField),
      volfracField_RK_agg(volfracField);

  Box3D domain(0, nx - 1, 0, ny - 1, 1, nz - 2);
  Box3D bottom(0, nx - 1, 0, ny - 1, 0, 0);
  Box3D bottom2(0, nx - 1, 0, ny - 1, 0, 1);
  Box3D top(0, nx - 1, 0, ny - 1, nz - 1, nz - 1);

  Box3D front(nx - 1, nx - 1, 0, ny - 1, 1, nz - 2);
  Box3D back(0, 0, 0, ny - 1, 1, nz - 2);

  Box3D left(0, nx - 1, 0, 0, 1, nz - 2);
  Box3D right(0, nx - 1, ny - 1, ny - 1, 1, nz - 2);

  Box3D no_bottom(0, nx - 1, 0, ny - 1, 2, nz - 1);

  std::vector<MultiBlock3D *> args_stp;
  args_stp.push_back(&volfracField_tot);
  args_stp.push_back(&volfracField);
  args_stp.push_back(&volfracField_agg);
  std::vector<MultiBlock3D *> args_vtk;
  args_vtk.push_back(&volfracField_tot);
  args_vtk.push_back(&volfracField);
  args_vtk.push_back(&Q);
  args_vtk.push_back(&v_sedimentation);
  args_vtk.push_back(&dissipation);
  args_vtk.push_back(&volfracField_agg);
  args_vtk.push_back(&Q_agg);

  nsLattice.periodicity().toggleAll(true);

  for (plint i = 0; i < 2; i++) {
    volfracField.periodicity().toggle(i, true);
    volfracField_tot.periodicity().toggle(i, true);
    Q.periodicity().toggle(i, true);
    densityField.periodicity().toggle(i, true);
    dissipation.periodicity().toggle(i, true);
    velocity.periodicity().toggle(i, true);
    D_t.periodicity().toggle(i, true);
    D_tp1.periodicity().toggle(i, true);
    Q_d.periodicity().toggle(i, true);
    v_sedimentation.periodicity().toggle(i, true);
    volfracField_RK.periodicity().toggle(i, true);
    phi_1.periodicity().toggle(i, true);
    phi_2.periodicity().toggle(i, true);
    phi_n_adv.periodicity().toggle(i, true);
    phi_1_adv.periodicity().toggle(i, true);
    phi_2_adv.periodicity().toggle(i, true);
    T_t.periodicity().toggle(i, true);

    volfracField_agg.periodicity().toggle(i, true);
    Q_agg.periodicity().toggle(i, true);
    v_sedimentation_agg.periodicity().toggle(i, true);
    volfracField_RK_agg.periodicity().toggle(i, true);
  }

  for (plint i = 0; i < 2; i++) {
    velX.periodicity().toggle(i, true);
    velY.periodicity().toggle(i, true);
    velZ.periodicity().toggle(i, true);
    Flux_px.periodicity().toggle(i, true);
    Flux_py.periodicity().toggle(i, true);
    Flux_pz.periodicity().toggle(i, true);
    Flux_nx.periodicity().toggle(i, true);
    Flux_ny.periodicity().toggle(i, true);
    Flux_nz.periodicity().toggle(i, true);
  }

  Dynamics<T, NSDESCRIPTOR> *nsbbDynamics = new BounceBack<T, NSDESCRIPTOR>();
  defineDynamics(nsLattice, bottom, nsbbDynamics->clone());
  defineDynamics(nsLattice, top, nsbbDynamics->clone());
  delete nsbbDynamics;
  nsbbDynamics = 0;

  nsLattice.toggleInternalStatistics(false);

  ExpSetup(nsLattice, args_stp, weights, TotalVolFrac, densityField, tmp,
           parameters);

  Array<T, NSDESCRIPTOR<T>::d> forceOrientation(T(), T(), (T)1);
  std::vector<MultiBlock3D *> args_f;
  args_f.push_back(&nsLattice);
  args_f.push_back(&densityField);
  args_f.push_back(&volfracField);
  args_f.push_back(&volfracField_agg);

  integrateProcessingFunctional(
      new ScalarBuoyanTermProcessor3D_sev<T, NSDESCRIPTOR>(
          g_lb, rhoUp_k, rhoPi_k, rhoPa_k, parameters.getDeltaT(),
          forceOrientation),
      nsLattice.getBoundingBox(), args_f, 1);

  T tIni = global::timer("simTime").stop();
  pcout << "time elapsed for ExpSetup:" << tIni << endl;
  global::timer("simTime").start();

  plint evalTime = 2000;
  plint iT = 0;
  plint maxT = 120 / parameters.getDeltaT();
  plint saveIter = 0.5 / parameters.getDeltaT();
  plint saveIterVtk = 0.5 / parameters.getDeltaT();
  plint saveCheck = 10 / parameters.getDeltaT();
  util::ValueTracer<T> converge((T)1, (T)100, 1.0e-3);

  pcout << "Max Number of iterations: " << maxT << endl;
  pcout << "Number of saving iterations: " << saveIter << endl;
  pcout << "Real viscosity =  "
        << parameters.getLatticeNu() *
               (parameters.getDeltaX() * parameters.getDeltaX()) /
               parameters.getDeltaT()
        << endl;

  T M_ini = ComputeTotalMass(volfracField, volfracField_agg, parameters);
  pcout << "Initial Mass = " << M_ini << endl;

  plb::copy(*multiply(volfracField, scale), volfracField_RK,
            volfracField.getBoundingBox());
  plb::copy(*multiply(volfracField_agg, scale), volfracField_RK_agg,
            volfracField_agg.getBoundingBox());

  for (iT = 0; iT <= maxT; ++iT) {
    if (iT == (evalTime)) {
      T tEval = global::timer("simTime").stop();
      T remainTime = (tEval - tIni) / (T)evalTime * (T)maxT / (T)3600;
      global::timer("simTime").start();
      pcout << "Remaining " << (plint)remainTime << " hours, and ";
      pcout << (plint)((T)60 * (remainTime - (T)((plint)remainTime)) + 0.5)
            << " minutes." << endl;
    }

    if (iT % saveIterVtk == 0) {
      pcout << iT * parameters.getDeltaT() << " : Writing VTK." << endl;
      writeVTK(nsLattice, args_vtk, parameters, iT);
      pcout << "Total Mass = "
            << ComputeTotalMass(volfracField, volfracField_agg, parameters)
            << endl;
      pcout << "Mass Out = "
            << M_ini -
                   ComputeTotalMass(volfracField, volfracField_agg, parameters)
            << endl;
    }

    if (iT % saveIter == 0) {
      pcout << iT * parameters.getDeltaT() << " : Writing gif." << endl;
      writeGif(nsLattice, volfracField_tot, densityField, iT);
    }

    if (iT % saveCheck == 0) {
      std::string directoryPath = ".";
      std::string pattern = "checkpoint_*";
      for (const auto &entry : filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file() && entry.path().filename().string().substr(
                                           0, pattern.size()) == pattern) {
          if (filesystem::remove(entry.path())) {
            std::cout << "Removed file: " << entry.path() << std::endl;
          } else {
            std::cerr << "Error: Unable to remove file: " << entry.path()
                      << std::endl;
          }
        }
      }
      pcout << "Saving checkpoints at iteration " << iT << endl;
      saveBinaryBlock(nsLattice, "checkpoint_fluid.dat");
      saveBinaryBlock(volfracField_tot, "checkpoint_vf_tot.dat");
      saveBinaryBlock(densityField, "checkpoint_dens.dat");
      saveBinaryBlock(volfracField, "checkpoint_vf.dat");
      saveBinaryBlock(volfracField_agg, "checkpoint_vf_agg.dat");
    }

    bool upwind = true;
    bool neumann = true;

    // Lattice Boltzmann iteration step.
    nsLattice.collideAndStream();

    computeVelocity(nsLattice, velocity, nsLattice.getBoundingBox());

    computeVelocityComponent(nsLattice, velX, nsLattice.getBoundingBox(), 0);
    computeVelocityComponent(nsLattice, velY, nsLattice.getBoundingBox(), 1);
    computeVelocityComponent(nsLattice, velZ, nsLattice.getBoundingBox(), 2);
    T max_f = computeMax(*computeAbsoluteValue(velZ));
    for (int iD = 0; iD < N; ++iD) {
      T max_sed = computeMax(
          *computeAbsoluteValue(*extractComponent(v_sedimentation, iD)));
      T max_sed_agg = computeMax(
          *computeAbsoluteValue(*extractComponent(v_sedimentation_agg, iD)));
      alpha_x[iD] = computeMax(*computeAbsoluteValue(velX));
      alpha_y[iD] = computeMax(*computeAbsoluteValue(velY));
      alpha_z[iD] = max_f + std::max(max_sed, max_sed_agg);
    }

    Actions3D cycle;

    plint nsLattice_ID = cycle.addBlock(nsLattice);
    plint D_t_ID = cycle.addBlock(D_t);
    plint D_tp1_ID = cycle.addBlock(D_tp1);
    plint densityField_ID1 = cycle.addBlock(densityField);
    plint velocity_ID1 = cycle.addBlock(velocity);
    plint Q_d_ID = cycle.addBlock(Q_d);
    plint densityField_ID = cycle.addBlock(densityField);
    plint T_t_ID = cycle.addBlock(T_t);
    plint dissipation_ID = cycle.addBlock(dissipation);
    plint volfracField_ID = cycle.addBlock(volfracField);
    plint volfracField_tot_ID = cycle.addBlock(volfracField_tot);
    plint volfracField_RK_ID = cycle.addBlock(volfracField_RK);
    plint v_sedimentation_ID = cycle.addBlock(v_sedimentation);
    plint Q_ID = cycle.addBlock(Q);
    plint phi_n_adv_ID = cycle.addBlock(phi_n_adv);
    plint phi_1_ID = cycle.addBlock(phi_1);
    plint phi_2_ID = cycle.addBlock(phi_2);
    plint phi_1_adv_ID = cycle.addBlock(phi_1_adv);
    plint phi_2_adv_ID = cycle.addBlock(phi_2_adv);

    plint volfracField_agg_ID = cycle.addBlock(volfracField_agg);
    plint v_sedimentation_agg_ID = cycle.addBlock(v_sedimentation_agg);
    plint Q_agg_ID = cycle.addBlock(Q_agg);
    plint volfracField_RK_agg_ID = cycle.addBlock(volfracField_RK_agg);

    plint Flux_px_ID = cycle.addBlock(Flux_px);
    plint Flux_py_ID = cycle.addBlock(Flux_py);
    plint Flux_pz_ID = cycle.addBlock(Flux_pz);
    plint Flux_nx_ID = cycle.addBlock(Flux_nx);
    plint Flux_ny_ID = cycle.addBlock(Flux_ny);
    plint Flux_nz_ID = cycle.addBlock(Flux_nz);

    cycle.addProcessor(new ComputeDissipationRate<T, NSDESCRIPTOR>(cSmago),
                       nsLattice_ID, dissipation_ID,
                       nsLattice.getBoundingBox());
    cycle.addCommunication(dissipation_ID, modif::staticVariables);
    cycle.addProcessor(new CopyConvertScalarFunctional3D<T, T>(),
                       densityField_ID1, D_t_ID, densityField.getBoundingBox());
    cycle.addCommunication(D_t_ID, modif::staticVariables);
    cycle.addProcessor(new CopyConvertScalarFunctional3D<T, T>(),
                       densityField_ID1, D_tp1_ID,
                       densityField.getBoundingBox());
    cycle.addCommunication(D_tp1_ID, modif::staticVariables);
    cycle.addProcessor(
        new AdvectionDiffusionFd3D_neumann<T>(
            Di_k * parameters.getLatticeKappa(), upwind, neumann, nx, ny, nz),
        D_t_ID, D_tp1_ID, densityField_ID1, velocity_ID1, Q_d_ID,
        densityField.getBoundingBox());
    cycle.addCommunication(densityField_ID1, modif::staticVariables);

    std::vector<plint> source_ID;
    source_ID.push_back(dissipation_ID);
    source_ID.push_back(volfracField_ID);
    source_ID.push_back(Q_ID);
    source_ID.push_back(volfracField_agg_ID);
    source_ID.push_back(Q_agg_ID);

    cycle.addProcessor(new BirthAndDeath<T, NSDESCRIPTOR>(
                           K_i, K_a, K_i_a, K_a_i, KTS_i, KTS_a, KTS_i_a,
                           parameters.getDeltaX(), parameters.getDeltaT(), Dp_k,
                           Dp_agg_k, m, coeff, rhoPi_k, rhoPa_k),
                       source_ID, no_bottom);
    cycle.addCommunication(Q_ID, modif::staticVariables);
    cycle.addCommunication(Q_agg_ID, modif::staticVariables);
    cycle.addProcessor(
        new Tensor_A_times_alpha_inplace_functional3D<T, N>(scale), Q_ID,
        Q.getBoundingBox());
    cycle.addCommunication(Q_ID, modif::staticVariables);
    cycle.addProcessor(
        new Tensor_A_times_alpha_inplace_functional3D<T, N>(scale), Q_agg_ID,
        Q_agg.getBoundingBox());
    cycle.addCommunication(Q_agg_ID, modif::staticVariables);

    std::vector<plint> arg1_1;
    std::vector<plint> arg1_2;
    std::vector<plint> arg1_3;

    std::vector<plint> vf_arg;
    vf_arg.push_back(volfracField_ID);
    vf_arg.push_back(volfracField_agg_ID);
    std::vector<plint> vf_arg_RK;
    vf_arg_RK.push_back(volfracField_RK_ID);
    vf_arg_RK.push_back(volfracField_RK_agg_ID);
    std::vector<plint> Q_arg;
    Q_arg.push_back(Q_ID);
    Q_arg.push_back(Q_agg_ID);
    std::vector<plint> vsed_arg;
    vsed_arg.push_back(v_sedimentation_ID);
    vsed_arg.push_back(v_sedimentation_agg_ID);

    for (int chk = 0; chk < 2; ++chk) {
      if (chk == 0) {
        rhoP_k = rhoPi_k;
        Dip_k = Dp_k;
      } else {
        rhoP_k = rhoPa_k;
        Dip_k = Dp_agg_k;
      }
      cycle.addProcessor(
          new Get_v_sedimentation<T>(rhoP_k, Dip_k, convers, mu_k, g_k),
          densityField_ID, vsed_arg[chk], v_sedimentation.getBoundingBox());
      cycle.addCommunication(vsed_arg[chk], modif::staticVariables);
      std::vector<plint> arg_flu;
      std::vector<plint> argF_1;
      std::vector<plint> arg_flu2;
      std::vector<plint> argF_2;
      std::vector<plint> arg_flu3;
      std::vector<plint> argF_3;
      arg_flu.push_back(vf_arg_RK[chk]);
      arg_flu.push_back(Flux_px_ID);
      arg_flu.push_back(Flux_py_ID);
      arg_flu.push_back(Flux_pz_ID);
      arg_flu.push_back(Flux_nx_ID);
      arg_flu.push_back(Flux_ny_ID);
      arg_flu.push_back(Flux_nz_ID);
      arg_flu.push_back(nsLattice_ID);
      arg_flu.push_back(vsed_arg[chk]);
      cycle.addProcessor(
          new ComputeFluxes<T, NSDESCRIPTOR>(alpha_x, alpha_y, alpha_z),
          arg_flu, volfracField.getBoundingBox());
      cycle.addCommunication(Flux_px_ID, modif::staticVariables);
      cycle.addCommunication(Flux_py_ID, modif::staticVariables);
      cycle.addCommunication(Flux_pz_ID, modif::staticVariables);
      cycle.addCommunication(Flux_nx_ID, modif::staticVariables);
      cycle.addCommunication(Flux_ny_ID, modif::staticVariables);
      cycle.addCommunication(Flux_nz_ID, modif::staticVariables);
      argF_1.push_back(Flux_px_ID);
      argF_1.push_back(Flux_py_ID);
      argF_1.push_back(Flux_pz_ID);
      argF_1.push_back(Flux_nx_ID);
      argF_1.push_back(Flux_ny_ID);
      argF_1.push_back(Flux_nz_ID);
      argF_1.push_back(phi_n_adv_ID);
      argF_1.push_back(Q_arg[chk]);
      cycle.addProcessor(new AdvectionDiffusionFd3D2_WENO3_f<T>(
                             Di_k * parameters.getLatticeKappa(), eps, neumann,
                             nx, ny, nz, alpha_x, alpha_y, alpha_z),
                         argF_1, volfracField_tot.getBoundingBox());
      cycle.addCommunication(phi_n_adv_ID, modif::staticVariables);
      cycle.addProcessor(new RK3_Step1_functional3D<T>(), vf_arg_RK[chk],
                         phi_n_adv_ID, phi_1_ID,
                         volfracField_tot.getBoundingBox());
      cycle.addCommunication(phi_1_ID, modif::staticVariables);
      cycle.addProcessor(new CopyConvertTensorFunctional3D<T, T, N>(), phi_1_ID,
                         T_t_ID, volfracField_tot.getBoundingBox());
      cycle.addCommunication(T_t_ID, modif::staticVariables);
      arg_flu2.push_back(T_t_ID);
      arg_flu2.push_back(Flux_px_ID);
      arg_flu2.push_back(Flux_py_ID);
      arg_flu2.push_back(Flux_pz_ID);
      arg_flu2.push_back(Flux_nx_ID);
      arg_flu2.push_back(Flux_ny_ID);
      arg_flu2.push_back(Flux_nz_ID);
      arg_flu2.push_back(nsLattice_ID);
      arg_flu2.push_back(vsed_arg[chk]);
      cycle.addProcessor(
          new ComputeFluxes<T, NSDESCRIPTOR>(alpha_x, alpha_y, alpha_z),
          arg_flu2, volfracField.getBoundingBox());
      cycle.addCommunication(Flux_px_ID, modif::staticVariables);
      cycle.addCommunication(Flux_py_ID, modif::staticVariables);
      cycle.addCommunication(Flux_pz_ID, modif::staticVariables);
      cycle.addCommunication(Flux_nx_ID, modif::staticVariables);
      cycle.addCommunication(Flux_ny_ID, modif::staticVariables);
      cycle.addCommunication(Flux_nz_ID, modif::staticVariables);
      argF_2.push_back(Flux_px_ID);
      argF_2.push_back(Flux_py_ID);
      argF_2.push_back(Flux_pz_ID);
      argF_2.push_back(Flux_nx_ID);
      argF_2.push_back(Flux_ny_ID);
      argF_2.push_back(Flux_nz_ID);
      argF_2.push_back(phi_1_adv_ID);
      argF_2.push_back(Q_arg[chk]);
      cycle.addProcessor(new AdvectionDiffusionFd3D2_WENO3_f<T>(
                             Di_k * parameters.getLatticeKappa(), eps, neumann,
                             nx, ny, nz, alpha_x, alpha_y, alpha_z),
                         argF_2, volfracField_tot.getBoundingBox());
      cycle.addCommunication(phi_1_adv_ID, modif::staticVariables);
      cycle.addProcessor(new RK3_Step2_functional3D<T>(), vf_arg_RK[chk],
                         phi_1_ID, phi_1_adv_ID, phi_2_ID,
                         volfracField_tot.getBoundingBox());
      cycle.addCommunication(phi_1_adv_ID, modif::staticVariables);

      cycle.addProcessor(new CopyConvertTensorFunctional3D<T, T, N>(), phi_2_ID,
                         T_t_ID, volfracField_tot.getBoundingBox());
      cycle.addCommunication(T_t_ID, modif::staticVariables);
      arg_flu3.push_back(T_t_ID);
      arg_flu3.push_back(Flux_px_ID);
      arg_flu3.push_back(Flux_py_ID);
      arg_flu3.push_back(Flux_pz_ID);
      arg_flu3.push_back(Flux_nx_ID);
      arg_flu3.push_back(Flux_ny_ID);
      arg_flu3.push_back(Flux_nz_ID);
      arg_flu3.push_back(nsLattice_ID);
      arg_flu3.push_back(vsed_arg[chk]);
      cycle.addProcessor(
          new ComputeFluxes<T, NSDESCRIPTOR>(alpha_x, alpha_y, alpha_z),
          arg_flu3, volfracField.getBoundingBox());
      cycle.addCommunication(Flux_px_ID, modif::staticVariables);
      cycle.addCommunication(Flux_py_ID, modif::staticVariables);
      cycle.addCommunication(Flux_pz_ID, modif::staticVariables);
      cycle.addCommunication(Flux_nx_ID, modif::staticVariables);
      cycle.addCommunication(Flux_ny_ID, modif::staticVariables);
      cycle.addCommunication(Flux_nz_ID, modif::staticVariables);
      argF_3.push_back(Flux_px_ID);
      argF_3.push_back(Flux_py_ID);
      argF_3.push_back(Flux_pz_ID);
      argF_3.push_back(Flux_nx_ID);
      argF_3.push_back(Flux_ny_ID);
      argF_3.push_back(Flux_nz_ID);
      argF_3.push_back(phi_2_adv_ID);
      argF_3.push_back(Q_arg[chk]);
      cycle.addProcessor(new AdvectionDiffusionFd3D2_WENO3_f<T>(
                             Di_k * parameters.getLatticeKappa(), eps, neumann,
                             nx, ny, nz, alpha_x, alpha_y, alpha_z),
                         argF_3, volfracField_tot.getBoundingBox());
      cycle.addCommunication(phi_2_adv_ID, modif::staticVariables);
      cycle.addProcessor(new RK3_Step3_functional3D<T>(), vf_arg_RK[chk],
                         phi_2_ID, phi_2_adv_ID, vf_arg_RK[chk],
                         volfracField_tot.getBoundingBox());
      cycle.addCommunication(vf_arg_RK[chk], modif::staticVariables);
      cycle.addProcessor(new CopyConvertTensorFunctional3D<T, T, N>(),
                         vf_arg_RK[chk], vf_arg[chk],
                         volfracField_tot.getBoundingBox());
      cycle.addCommunication(vf_arg[chk], modif::staticVariables);
      cycle.addProcessor(
          new Tensor_A_times_alpha_inplace_functional3D<T, N>(1 / scale),
          vf_arg[chk], volfracField_tot.getBoundingBox());
      cycle.addCommunication(vf_arg[chk], modif::staticVariables);
      arg_flu.clear();
      arg_flu2.clear();
      arg_flu3.clear();
      argF_1.clear();
      argF_2.clear();
      argF_3.clear();
      vf_arg.clear();
      vf_arg_RK.clear();
      Q_arg.clear();
      vsed_arg.clear();
    }

    /////////////////////
    // Regularization ///
    /////////////////////
    cycle.addProcessor(new Regu_VF_functional3D<T>(), volfracField_tot_ID,
                       volfracField_ID, volfracField_agg_ID,
                       volfracField_tot.getBoundingBox());
    cycle.addCommunication(volfracField_tot_ID, modif::staticVariables);

    cycle.execute();
  }

  writeGif(nsLattice, volfracField_tot, densityField, iT);

  T tEnd = global::timer("simTime").stop();

  T totalTime = tEnd - tIni;
  T nx100 = nsLattice.getNx() / (T)100;
  T ny100 = nsLattice.getNy() / (T)100;
  T nz100 = nsLattice.getNz() / (T)100;
  pcout << "N=" << resolution << endl;
  pcout << "number of processors: " << global::mpi().getSize() << endl;
  pcout << "simulation time: " << totalTime << endl;
  pcout << "total time: " << tEnd << endl;
  pcout << "total iterations: " << iT << endl;
  pcout << "Msus: " << nx100 * ny100 * nz100 * (T)iT / totalTime << endl;

  return 0;
}
