#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "laplace_operator.h"

namespace NavierStokes_DG
{
  using namespace dealii;

  constexpr unsigned int dimension     = 3;
  constexpr unsigned int fe_degree     = 4;
  constexpr unsigned int n_q_points_1d = fe_degree + 2;

  constexpr unsigned int group_size = numbers::invalid_unsigned_int;

  using Number = double;

  using VectorizedArrayType = VectorizedArray<Number>;

  struct Parameters
  {
    Parameters(const std::string &filename)
      : gamma(1)
      , R(0)
      , c_v(0)
      , c_p(0)
      , viscosity(0)
      , lambda(0)
      , Ma(0)
      , output_tick(0)
      , refine_tick(0)
      , courant_number(0)
      , n_refinements(0)
      , end_time(0)
      , print_debug_timings(false)
    {
      ParameterHandler prm;
      {
        prm.enter_subsection("Flow parameters");
        prm.add_parameter("gamma",
                          gamma,
                          "Gas constant for ideal gas",
                          Patterns::Double(1),
                          true);
        prm.add_parameter(
          "R", R, "Specific gas constant", Patterns::Double(0), true);
        prm.add_parameter("c_v",
                          c_v,
                          "Specific heat capacity at constant volume",
                          Patterns::Double(0),
                          true);
        prm.add_parameter("c_p",
                          c_p,
                          "Specific heat capacity at constant pressure",
                          Patterns::Double(0),
                          true);
        prm.add_parameter("viscosity",
                          viscosity,
                          "Fluid dynamic viscosity",
                          Patterns::Double(0),
                          true);
        prm.add_parameter(
          "lambda", lambda, "Thermal conductivity", Patterns::Double(0), true);
        prm.add_parameter(
          "Mach number", Ma, "Mach number", Patterns::Double(0), true);
        prm.leave_subsection();
      }
      {
        prm.enter_subsection("Control parameters");
        prm.add_parameter("output tick",
                          output_tick,
                          "Define at which time interval to create output",
                          Patterns::Double(0),
                          true);
        prm.add_parameter(
          "refine tick",
          refine_tick,
          "Define at which time interval to dynamically adapt the mesh",
          Patterns::Double(0),
          true);

        prm.add_parameter(
          "Courant number",
          courant_number,
          "Courant number controlling the time step via dt = Courant * h",
          Patterns::Double(0),
          true);
        prm.add_parameter(
          "refinements",
          n_refinements,
          "Number of refinements, resulting in 64 * 2^refinements elements",
          Patterns::Integer(0),
          true);
        prm.add_parameter("end time",
                          end_time,
                          "End time for the simulation",
                          Patterns::Double(0),
                          true);
        prm.add_parameter("print debug timings",
                          print_debug_timings,
                          "true/false",
                          Patterns::Bool(),
                          true);
        prm.leave_subsection();
      }
      prm.parse_input(filename, "", true, true);
    }

    double       gamma;
    double       R;
    double       c_v;
    double       c_p;
    double       viscosity;
    double       lambda;
    double       Ma;
    double       output_tick;
    double       refine_tick;
    double       courant_number;
    unsigned int n_refinements;
    double       end_time;
    bool         print_debug_timings;
  };



  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    InitialCondition(const double time, const Parameters &param)
      : Function<dim>(dim + 2, time)
      , parameters(param)
    {}

    virtual double
    value(const Point<dim> &xx, const unsigned int component = 0) const override
    {
      Point<3> x;
      for (unsigned int d = 0; d < dim; ++d)
        x[d] = xx[d];
      const double c0 = 1. / parameters.Ma;
      const double T0 = c0 * c0 / (parameters.gamma * parameters.R);
      if (component == 0)
        return 1 + 1. / (parameters.R * T0) * 1. / 16. *
                     (std::cos(2 * x[0]) + std::cos(2 * x[1])) *
                     (std::cos(2 * x[2]) + 2.);
      else if (component == 1)
        return std::sin(x[0]) * std::cos(x[1]) * std::cos(x[2]);
      else if (component == 2)
        return -std::cos(x[0]) * std::sin(x[1]) * std::cos(x[2]);
      else if (component == dim)
        return 0.;
      else
        return parameters.c_v * T0 +
               0.5 * (Utilities::fixed_power<2>(
                        std::sin(x[0]) * std::cos(x[1]) * std::cos(x[2])) +
                      Utilities::fixed_power<2>(
                        std::cos(x[0]) * std::sin(x[1]) * std::cos(x[2])));
    }

  private:
    const Parameters &parameters;
  };



  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    euler_velocity(const Tensor<1, dim + 2, Number> &conserved_variables)
  {
    const Number inverse_density = Number(1.) / conserved_variables[0];

    Tensor<1, dim, Number> velocity;
    for (unsigned int d = 0; d < dim; ++d)
      velocity[d] = conserved_variables[1 + d] * inverse_density;

    return velocity;
  }



  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    euler_pressure(const Tensor<1, dim + 2, Number> &conserved_variables,
                   const Parameters &                parameters)
  {
    const Tensor<1, dim, Number> velocity =
      euler_velocity<dim>(conserved_variables);

    Number rho_u_dot_u = conserved_variables[1] * velocity[0];
    for (unsigned int d = 1; d < dim; ++d)
      rho_u_dot_u += conserved_variables[1 + d] * velocity[d];

    return (parameters.gamma - 1.) *
           (conserved_variables[dim + 1] - 0.5 * rho_u_dot_u);
  }



  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim + 2, Tensor<1, dim, Number>>
    euler_flux(const Tensor<1, dim + 2, Number> &conserved_variables,
               const Parameters &                parameters)
  {
    const Tensor<1, dim, Number> velocity =
      euler_velocity<dim>(conserved_variables);
    const Number pressure =
      euler_pressure<dim>(conserved_variables, parameters);

    Tensor<1, dim + 2, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      {
        flux[0][d] = conserved_variables[1 + d];
        for (unsigned int e = 0; e < dim; ++e)
          flux[e + 1][d] = conserved_variables[e + 1] * velocity[d];
        flux[d + 1][d] += pressure;
        flux[dim + 1][d] =
          velocity[d] * (conserved_variables[dim + 1] + pressure);
      }

    return flux;
  }



  template <int n_components, int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_components, Number>
    operator*(const Tensor<1, n_components, Tensor<1, dim, Number>> &matrix,
              const Tensor<1, dim, Number> &                         vector)
  {
    Tensor<1, n_components, Number> result;
    for (unsigned int d = 0; d < n_components; ++d)
      result[d] = matrix[d] * vector;
    return result;
  }



  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim + 2, Number>
    euler_numerical_flux(const Tensor<1, dim + 2, Number> &u_m,
                         const Tensor<1, dim + 2, Number> &u_p,
                         const Tensor<1, dim, Number> &    normal,
                         const Parameters &                parameters)
  {
    const auto velocity_m = euler_velocity<dim>(u_m);
    const auto velocity_p = euler_velocity<dim>(u_p);

    const auto pressure_m = euler_pressure<dim>(u_m, parameters);
    const auto pressure_p = euler_pressure<dim>(u_p, parameters);

    const auto flux_m = euler_flux<dim>(u_m, parameters);
    const auto flux_p = euler_flux<dim>(u_p, parameters);

    const auto Lambda =
      0.5 *
      std::sqrt(std::max(velocity_p.norm_square() +
                           parameters.gamma * pressure_p * (1. / u_p[0]),
                         velocity_m.norm_square() +
                           parameters.gamma * pressure_m * (1. / u_m[0])));

    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * Lambda * (u_m - u_p);
  }



  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, Number>
    velocity_gradient(
      const Tensor<1, dim + 2, Number> &                conserved_variables,
      const Tensor<1, dim + 2, Tensor<1, dim, Number>> &gradients)
  {
    const Number inverse_density = Number(1.) / conserved_variables[0];
    Tensor<2, dim, Number> result;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const Number ud = inverse_density * conserved_variables[d + 1];
        for (unsigned int e = 0; e < dim; ++e)
          result[d][e] =
            inverse_density * (gradients[d + 1][e] - ud * gradients[0][e]);
      }

    return result;
  }



  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    temperature(const Tensor<1, dim + 2, Number> &conserved_variables,
                const Parameters &                parameters)
  {
    const Number inverse_density = Number(1.) / conserved_variables[0];
    const Number inverse_R       = 1. / parameters.R;
    return euler_pressure(conserved_variables) * inverse_density * inverse_R;
  }



  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    temperature_gradient(
      const Tensor<1, dim + 2, Number> &                conserved_variables,
      const Tensor<1, dim + 2, Tensor<1, dim, Number>> &gradients,
      const Parameters &                                parameters)
  {
    const Number inverse_R = 1. / parameters.R;
    return (parameters.gamma - 1.) * inverse_R *
           (gradients[dim + 1] -
            euler_velocity<dim>(conserved_variables) *
              velocity_gradient(conserved_variables, gradients));
  }



  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<2, dim, Number>
    viscous_flux(const Tensor<1, dim + 2, Number> &conserved_variables,
                 const Tensor<1, dim + 2, Tensor<1, dim, Number>> &gradients,
                 const Parameters &                                parameters)
  {
    const Tensor<2, dim, Number> grad_u =
      velocity_gradient(conserved_variables, gradients);
    const Number scaled_div_u =
      parameters.viscosity * (2. / 3.) * trace(grad_u);

    Tensor<2, dim, Number> result;
    for (unsigned int d = 0; d < dim; ++d)
      {
        for (unsigned int e = d; e < dim; ++e)
          {
            result[d][e] = parameters.viscosity * (grad_u[d][e] + grad_u[e][d]);
            result[e][d] = result[d][e];
          }
        result[d][d] -= scaled_div_u;
      }
    return result;
  }



  template <int dim, typename Number>
  VectorizedArray<Number>
  evaluate_function(const Function<dim> &                      function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized,
                    const unsigned int                         component)
  {
    VectorizedArray<Number> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        result[v] = function.value(p, component);
      }
    return result;
  }



  template <int dim, typename Number, int n_components = dim + 2>
  Tensor<1, n_components, VectorizedArray<Number>>
  evaluate_function(const Function<dim> &                      function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized)
  {
    AssertDimension(function.n_components, n_components);
    Tensor<1, n_components, VectorizedArray<Number>> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        for (unsigned int d = 0; d < n_components; ++d)
          result[d][v] = function.value(p, d);
      }
    return result;
  }



  template <int dim, int degree, int n_points_1d>
  class EulerOperator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    EulerOperator(TimerOutput &timer_output, const Parameters &parameters);

    ~EulerOperator();

    void
    reinit(const Mapping<dim> &       mapping,
           const DoFHandler<dim> &    dof_handler,
           DoFHandler<dim> &          dof_handler_velocity,
           AffineConstraints<Number> &constraints_velocity,
           DoFHandler<dim> &          dof_handler_energy,
           AffineConstraints<Number> &constraints_energy);

    void
    set_inflow_boundary(const types::boundary_id       boundary_id,
                        std::unique_ptr<Function<dim>> inflow_function);

    void
    set_subsonic_outflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> outflow_energy);

    void
    set_wall_boundary(const types::boundary_id boundary_id);

    void
    apply(const double                                      current_time,
          const LinearAlgebra::distributed::Vector<Number> &src,
          LinearAlgebra::distributed::Vector<Number> &      dst) const;

    void
    project(const Function<dim> &                       function,
            LinearAlgebra::distributed::Vector<Number> &solution) const;

    std::array<double, 2>
    compute_kinetic_energy(
      const LinearAlgebra::distributed::Vector<Number> &solution) const;

    double
    compute_cell_transport_speed(
      const LinearAlgebra::distributed::Vector<Number> &solution) const;

    void
    initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector) const;

    const MatrixFree<dim, Number, VectorizedArrayType> &
    get_matrix_free() const
    {
      return data;
    }

  private:
    MPI_Comm subcommunicator;

    MatrixFree<dim, Number, VectorizedArrayType> data;

    TimerOutput &     timer;
    const Parameters &parameters;

    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
      inflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   subsonic_outflow_boundaries;
    std::set<types::boundary_id>   wall_boundaries;
    std::unique_ptr<Function<dim>> body_force;

    void
    compute_constraints(const DoFHandler<dim> &    dof_handler,
                        AffineConstraints<Number> &constraints) const;

    void
    local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void
    local_apply_cell(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void
    local_apply_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     face_range) const;

    void
    local_apply_boundary_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     face_range) const;
  };



  template <int dim, int degree, int n_points_1d>
  EulerOperator<dim, degree, n_points_1d>::EulerOperator(
    TimerOutput &     timer,
    const Parameters &parameters)
    : timer(timer)
    , parameters(parameters)
  {
#ifdef DEAL_II_WITH_MPI
    if (group_size == 1)
      {
        this->subcommunicator = MPI_COMM_SELF;
      }
    else if (group_size == numbers::invalid_unsigned_int)
      {
        const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

        MPI_Comm_split_type(MPI_COMM_WORLD,
                            MPI_COMM_TYPE_SHARED,
                            rank,
                            MPI_INFO_NULL,
                            &subcommunicator);
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
#else
    (void)subcommunicator;
    (void)group_size;
    this->subcommunicator = MPI_COMM_SELF;
#endif
  }



  template <int dim, int degree, int n_points_1d>
  EulerOperator<dim, degree, n_points_1d>::~EulerOperator()
  {
#ifdef DEAL_II_WITH_MPI
    if (this->subcommunicator != MPI_COMM_SELF)
      MPI_Comm_free(&subcommunicator);
#endif
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &       mapping,
    const DoFHandler<dim> &    dof_handler,
    DoFHandler<dim> &          dof_handler_velocity,
    AffineConstraints<Number> &constraints_velocity,
    DoFHandler<dim> &          dof_handler_energy,
    AffineConstraints<Number> &constraints_energy)
  {
    compute_constraints(dof_handler_velocity, constraints_velocity);
    compute_constraints(dof_handler_energy, constraints_energy);

    const std::vector<const DoFHandler<dim> *> dof_handlers = {
      &dof_handler, &dof_handler_velocity, &dof_handler_energy};
    const AffineConstraints<double>                      dummy;
    const std::vector<const AffineConstraints<double> *> constraints = {
      &dummy, &constraints_velocity, &constraints_energy};
    const std::vector<Quadrature<1>> quadratures = {QGauss<1>(n_q_points_1d),
                                                    QGauss<1>(fe_degree + 1)};

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.mapping_update_flags_faces_by_cells =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData::none;

    additional_data.communicator_sm    = subcommunicator;
    additional_data.initialize_mapping = false;

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);

    DoFRenumbering::matrix_free_data_locality(dof_handler_velocity, data);
    DoFRenumbering::matrix_free_data_locality(dof_handler_energy, data);

    compute_constraints(dof_handler_velocity, constraints_velocity);
    compute_constraints(dof_handler_energy, constraints_energy);
    additional_data.initialize_mapping = true;
    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::compute_constraints(
    const DoFHandler<dim> &    dof_handler,
    AffineConstraints<Number> &constraints) const
  {
    constraints.clear();
    const IndexSet relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    constraints.reinit(relevant_dofs);
    std::vector<
      GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
      periodic_faces;
    for (unsigned int d = 0; d < dim; ++d)
      GridTools::collect_periodic_faces(
        dof_handler, 2 * d, 2 * d + 1, d, periodic_faces);
    DoFTools::make_periodicity_constraints<dim, dim>(periodic_faces,
                                                     constraints);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const
  {
    data.initialize_dof_vector(vector);
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::set_inflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> inflow_function)
  {
    AssertThrow(subsonic_outflow_boundaries.find(boundary_id) ==
                    subsonic_outflow_boundaries.end() &&
                  wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as inflow"));
    AssertThrow(inflow_function->n_components == dim + 2,
                ExcMessage("Expected function with dim+2 components"));

    inflow_boundaries[boundary_id] = std::move(inflow_function);
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::set_subsonic_outflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> outflow_function)
  {
    AssertThrow(inflow_boundaries.find(boundary_id) ==
                    inflow_boundaries.end() &&
                  wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as subsonic outflow"));
    AssertThrow(outflow_function->n_components == dim + 2,
                ExcMessage("Expected function with dim+2 components"));

    subsonic_outflow_boundaries[boundary_id] = std::move(outflow_function);
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::set_wall_boundary(
    const types::boundary_id boundary_id)
  {
    AssertThrow(inflow_boundaries.find(boundary_id) ==
                    inflow_boundaries.end() &&
                  subsonic_outflow_boundaries.find(boundary_id) ==
                    subsonic_outflow_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as wall boundary"));

    wall_boundaries.insert(boundary_id);
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::local_apply_cell(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto w_q = phi.get_value(q);
            phi.submit_gradient(euler_flux<dim>(w_q, parameters), q);
          }

        phi.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::local_apply_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi_m(data,
                                                                      true);
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi_p(data,
                                                                      false);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, EvaluationFlags::values);

        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
          {
            const auto numerical_flux =
              euler_numerical_flux<dim>(phi_m.get_value(q),
                                        phi_p.get_value(q),
                                        phi_m.get_normal_vector(q),
                                        parameters);
            phi_m.submit_value(-numerical_flux, q);
            phi_p.submit_value(numerical_flux, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi(data, true);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto w_m    = phi.get_value(q);
            const auto normal = phi.get_normal_vector(q);

            auto rho_u_dot_n = w_m[1] * normal[0];
            for (unsigned int d = 1; d < dim; ++d)
              rho_u_dot_n += w_m[1 + d] * normal[d];

            bool at_outflow = false;

            Tensor<1, dim + 2, VectorizedArray<Number>> w_p;
            const auto boundary_id = data.get_boundary_id(face);
            if (wall_boundaries.find(boundary_id) != wall_boundaries.end())
              {
                w_p[0] = w_m[0];
                for (unsigned int d = 0; d < dim; ++d)
                  w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
                w_p[dim + 1] = w_m[dim + 1];
              }
            else if (inflow_boundaries.find(boundary_id) !=
                     inflow_boundaries.end())
              w_p =
                evaluate_function(*inflow_boundaries.find(boundary_id)->second,
                                  phi.quadrature_point(q));
            else if (subsonic_outflow_boundaries.find(boundary_id) !=
                     subsonic_outflow_boundaries.end())
              {
                w_p          = w_m;
                w_p[dim + 1] = evaluate_function(
                  *subsonic_outflow_boundaries.find(boundary_id)->second,
                  phi.quadrature_point(q),
                  dim + 1);
                at_outflow = true;
              }
            else
              AssertThrow(false,
                          ExcMessage("Unknown boundary id, did "
                                     "you set a boundary condition for "
                                     "this part of the domain boundary?"));

            auto flux = euler_numerical_flux<dim>(w_m, w_p, normal, parameters);

            if (at_outflow)
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  if (rho_u_dot_n[v] < -1e-12)
                    for (unsigned int d = 0; d < dim; ++d)
                      flux[d + 1][v] = 0.;
                }

            phi.submit_value(-flux, q);
          }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 2, Number>
      inverse(phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
      }
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::apply(
    const double                                      current_time,
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      dst) const
  {
    {
      TimerOutput::Scope t(timer, "apply - integrals");

      for (auto &i : inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : subsonic_outflow_boundaries)
        i.second->set_time(current_time);

      data.loop(&EulerOperator::local_apply_cell,
                &EulerOperator::local_apply_face,
                &EulerOperator::local_apply_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
      TimerOutput::Scope t(timer, "apply - inverse mass");

      data.cell_loop(&EulerOperator::local_apply_inverse_mass_matrix,
                     this,
                     dst,
                     dst);
    }
  }



  template <int dim, int degree, int n_points_1d>
  void
  EulerOperator<dim, degree, n_points_1d>::project(
    const Function<dim> &                       function,
    LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number, VectorizedArrayType>
      phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim,
                                                   degree,
                                                   dim + 2,
                                                   Number,
                                                   VectorizedArrayType>
      inverse(phi);
    solution.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_dof_value(evaluate_function(function,
                                                 phi.quadrature_point(q)),
                               q);
        inverse.transform_from_q_points_to_basis(dim + 2,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(solution);
      }
  }



  template <int dim, int degree, int n_points_1d>
  std::array<double, 2>
  EulerOperator<dim, degree, n_points_1d>::compute_kinetic_energy(
    const LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    TimerOutput::Scope t(timer, "compute kinetic energy");
    double             squared[2] = {};
    FEEvaluation<dim, degree, n_points_1d, dim + 2, Number, VectorizedArrayType>
      phi(data, 0, 0);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution,
                            EvaluationFlags::values |
                              EvaluationFlags::gradients);
        VectorizedArrayType local_squared[2] = {};
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto JxW      = phi.JxW(q);
            const auto w_q      = phi.get_value(q);
            const auto velocity = euler_velocity<dim>(w_q);
            const auto velocity_grad =
              velocity_gradient(w_q, phi.get_gradient(q));
            local_squared[0] += velocity.norm_square() * JxW;
            local_squared[1] +=
              scalar_product(velocity_grad, velocity_grad) * JxW;
          }
        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          for (unsigned int d = 0; d < 2; ++d)
            squared[d] += local_squared[d][v];
      }

    Utilities::MPI::sum(squared, MPI_COMM_WORLD, squared);

    std::array<double, 2> result{
      {0.5 * squared[0] / Utilities::fixed_power<dim>(2. * numbers::PI),
       parameters.viscosity * squared[1] /
         Utilities::fixed_power<dim>(2. * numbers::PI)}};

    return result;
  }



  template <int dim, int degree, int n_points_1d>
  double
  EulerOperator<dim, degree, n_points_1d>::compute_cell_transport_speed(
    const LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    TimerOutput::Scope t(timer, "compute transport speed");
    Number             max_transport = 0;
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number, VectorizedArrayType>
      phi(data, 0, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArrayType local_max = 0.;
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto solution = phi.get_value(q);
            const auto velocity = euler_velocity<dim>(solution);
            const auto pressure = euler_pressure<dim>(solution, parameters);

            const auto          inverse_jacobian = phi.inverse_jacobian(q);
            const auto          convective_speed = inverse_jacobian * velocity;
            VectorizedArrayType convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              convective_limit =
                std::max(convective_limit, std::abs(convective_speed[d]));

            const auto speed_of_sound =
              std::sqrt(parameters.gamma * pressure * (1. / solution[0]));

            Tensor<1, dim, VectorizedArrayType> eigenvector;
            for (unsigned int d = 0; d < dim; ++d)
              eigenvector[d] = 1.;
            for (unsigned int i = 0; i < 5; ++i)
              {
                eigenvector = transpose(inverse_jacobian) *
                              (inverse_jacobian * eigenvector);
                VectorizedArrayType eigenvector_norm = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                  eigenvector_norm =
                    std::max(eigenvector_norm, std::abs(eigenvector[d]));
                eigenvector /= eigenvector_norm;
              }
            const auto jac_times_ev   = inverse_jacobian * eigenvector;
            const auto max_eigenvalue = std::sqrt(
              (jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
            local_max =
              std::max(local_max,
                       max_eigenvalue * speed_of_sound + convective_limit);
          }

        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          for (unsigned int d = 0; d < 3; ++d)
            max_transport = std::max(max_transport, local_max[v]);
      }

    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

    return max_transport;
  }



  template <int dim, int degree>
  class ViscousOperator
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    struct VelocityKernel
      : public Laplace::KernelBase<dim, dim, VectorizedArrayType>
    {
      VelocityKernel(const Parameters &parameters)
        : parameters(parameters)
      {}

      void
      set_density(const MatrixFree<dim, Number, VectorizedArrayType> &mf,
                  const VectorType &euler_solution)
      {
        FEEvaluation<dim, -1, 0, 1, Number> eval_density(mf, 0, 1, 0);
        if (densities.size(0) != mf.n_cell_batches())
          densities.reinit(mf.n_cell_batches(), eval_density.n_q_points);
        for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
          {
            eval_density.reinit(cell);
            eval_density.gather_evaluate(euler_solution,
                                         EvaluationFlags::values);
            for (unsigned int q : eval_density.quadrature_point_indices())
              densities(cell, q) = eval_density.get_value(q);
          }
      }

      virtual void
      pointwise_apply(Tensor<1, dim, VectorizedArrayType> &value,
                      Tensor<1, dim, Tensor<1, dim, VectorizedArrayType>> &grad,
                      const unsigned int                                   cell,
                      const unsigned int q_index) const override
      {
        Tensor<2, dim, VectorizedArrayType> viscous_stress;
        const VectorizedArrayType           scaled_div_u =
          (0.5 * time_step * parameters.viscosity * (2. / 3.)) *
          trace(Tensor<2, dim, VectorizedArrayType>(grad));
        for (unsigned int d = 0; d < dim; ++d)
          {
            for (unsigned int e = d; e < dim; ++e)
              {
                viscous_stress[d][e] =
                  (0.5 * time_step * parameters.viscosity) *
                  (grad[d][e] + grad[e][d]);
                viscous_stress[e][d] = viscous_stress[d][e];
              }
            viscous_stress[d][d] -= scaled_div_u;
          }
        grad = viscous_stress;
        for (unsigned int d = 0; d < dim; ++d)
          value[d] *= densities(cell, q_index);
      }

      Table<2, VectorizedArrayType> densities;
      double                        time_step;
      const Parameters &            parameters;
    };

    struct EnergyKernel
      : public Laplace::KernelBase<dim, 1, VectorizedArrayType>
    {
      EnergyKernel(const Table<2, VectorizedArrayType> &densities,
                   const Parameters &                   parameters)
        : densities(densities)
        , parameters(parameters)
      {}

      virtual void
      pointwise_apply(Tensor<1, 1, VectorizedArrayType> &                value,
                      Tensor<1, 1, Tensor<1, dim, VectorizedArrayType>> &grad,
                      const unsigned int                                 cell,
                      const unsigned int q_index) const override
      {
        grad = (time_step * parameters.c_v * parameters.lambda) * grad;
        value *= densities(cell, q_index);
      }

      const Table<2, VectorizedArrayType> &densities;
      double                               time_step;
      const Parameters &                   parameters;
    };

    using MatrixFreeType = MatrixFree<dim, Number, VectorizedArrayType>;
    const MatrixFreeType &data;

    const Parameters &parameters;

    VelocityKernel velocity_kernel;
    EnergyKernel   energy_kernel;

    Laplace::LaplaceOperator<dim, fe_degree, fe_degree + 1, dim>
      velocity_operator;
    Laplace::LaplaceOperator<dim, fe_degree, fe_degree + 1, 1> energy_operator;
    VectorType                 mass_diagonal_velocity;
    VectorType                 stiff_diagonal_velocity;
    DiagonalMatrix<VectorType> energy_diagonal;
    mutable double             velocity_solver_residuals;
    mutable unsigned int       n_velocity_solves;
    mutable double             energy_solver_residuals;

  public:
    VectorType     velocity_solution;
    VectorType     energy_solution;
    mutable double time_step;

    ViscousOperator(const MatrixFreeType &data, const Parameters &parameters)
      : data(data)
      , parameters(parameters)
      , velocity_kernel(parameters)
      , energy_kernel(velocity_kernel.densities, parameters)
      , velocity_operator(velocity_kernel)
      , energy_operator(energy_kernel)
      , velocity_solver_residuals(0)
      , n_velocity_solves(0)
      , energy_solver_residuals(0)
    {}

    void
    initialize()
    {
      velocity_operator.initialize(data, 1, 1);
      energy_operator.initialize(data, 2, 1);
      // ensure that the coefficient due to time stepping is 1 when it gets
      // eventually applied
      VectorType vector;
      data.initialize_dof_vector(vector);
      velocity_kernel.set_density(data, vector);
      velocity_kernel.time_step = 2.;
      stiff_diagonal_velocity   = velocity_operator.compute_stiff_diagonal();
      data.initialize_dof_vector(velocity_solution, 1);
      data.initialize_dof_vector(energy_solution, 2);
    }

    void
    propagate(const double time_step, VectorType &solution)
    {
      this->time_step = time_step;
      {
        DiagonalMatrix<VectorType> velocity_diagonal;
        velocity_diagonal.get_vector() =
          velocity_operator.compute_mass_diagonal(solution);
        velocity_diagonal.get_vector().add(time_step, stiff_diagonal_velocity);
        for (Number &a : velocity_diagonal.get_vector())
          if (a != 0.)
            a = 1. / a;
          else
            a = 1.;

        VectorType rhs;
        rhs.reinit(velocity_solution, true);
        data.loop(&ViscousOperator::velocity_rhs_cell,
                  &ViscousOperator::velocity_rhs_face,
                  &ViscousOperator::velocity_rhs_boundary,
                  this,
                  rhs,
                  solution,
                  true);
        velocity_kernel.set_density(data, solution);
        velocity_kernel.time_step = time_step;

        IterationNumberControl control(6, 1e-13);
        SolverCG<VectorType>   solver_cg(control);
        solver_cg.solve(velocity_operator,
                        velocity_solution,
                        rhs,
                        velocity_diagonal);
        ++n_velocity_solves;
        velocity_solver_residuals += control.last_value();
      }
      {
        VectorType rhs;
        rhs.reinit(energy_solution, true);
        velocity_solution.update_ghost_values();
        data.cell_loop(&ViscousOperator::energy_rhs, this, rhs, solution, true);
        DiagonalMatrix<VectorType> energy_diagonal;
        energy_diagonal.get_vector() =
          energy_operator.compute_mass_diagonal(solution);
        for (Number &a : energy_diagonal.get_vector())
          if (a != 0.)
            a = 1. / a;
          else
            a = 1.;

        energy_kernel.time_step = time_step;

        IterationNumberControl control(5, 1e-13);
        SolverCG<VectorType>   solver_cg(control);
        energy_solution = 0;
        solver_cg.solve(energy_operator, energy_solution, rhs, energy_diagonal);
        energy_solver_residuals += control.last_value();
      }
      data.cell_loop(&ViscousOperator::update_velocity,
                     this,
                     solution,
                     velocity_solution);

      // The case has low diffusivity, so simply ignore the energy update and
      // treat it as it were hyperbolic
    }

    void
    print_solver_statistics()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Velocity solver average converged residuals: "
                  << std::setprecision(3)
                  << static_cast<double>(velocity_solver_residuals) /
                       n_velocity_solves
                  << std::endl
                  << "Energy solver average converged residuals: "
                  << static_cast<double>(energy_solver_residuals) /
                       n_velocity_solves
                  << std::endl;
    }

  private:
    void
    velocity_rhs_cell(const MatrixFreeType &                       data,
                      VectorType &                                 dst,
                      const VectorType &                           src,
                      const std::pair<unsigned int, unsigned int> &cell_range)
    {
      FEEvaluation<dim, -1, 0, dim + 2, Number> eval_euler(data);
      FEEvaluation<dim, -1, 0, dim, Number>     eval_vel(data, 1);
      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          eval_euler.reinit(cell);
          eval_vel.reinit(cell);
          eval_euler.gather_evaluate(src,
                                     EvaluationFlags::values |
                                       EvaluationFlags::gradients);
          for (unsigned int q : eval_euler.quadrature_point_indices())
            {
              const auto value    = eval_euler.get_value(q);
              const auto grad     = eval_euler.get_gradient(q);
              const auto vel_flux = viscous_flux(value, grad, parameters);
              eval_vel.submit_gradient(Number(-0.5 * time_step) * vel_flux, q);
              Tensor<1, dim, VectorizedArrayType> momentum;
              for (unsigned int d = 0; d < dim; ++d)
                momentum[d] = value[d + 1];
              eval_vel.submit_value(momentum, q);
            }

          eval_vel.integrate_scatter(EvaluationFlags::values |
                                       EvaluationFlags::gradients,
                                     dst);
        }
    }

    void
    velocity_rhs_face(const MatrixFreeType &                       data,
                      VectorType &                                 dst,
                      const VectorType &                           src,
                      const std::pair<unsigned int, unsigned int> &face_range)
    {
      using FEFaceIntegrator    = FEFaceEvaluation<dim,
                                                degree,
                                                n_q_points_1d,
                                                dim + 2,
                                                Number,
                                                VectorizedArrayType>;
      using FEFaceIntegratorVel = FEFaceEvaluation<dim,
                                                   degree,
                                                   n_q_points_1d,
                                                   dim,
                                                   Number,
                                                   VectorizedArrayType>;
      FEFaceIntegrator    eval_m(data, true);
      FEFaceIntegrator    eval_p(data, false);
      FEFaceIntegratorVel eval_vel_m(data, true, 1);
      FEFaceIntegratorVel eval_vel_p(data, false, 1);
      for (unsigned int face = face_range.first; face < face_range.second;
           ++face)
        {
          eval_vel_m.reinit(face);
          eval_vel_p.reinit(face);
          eval_p.reinit(face);
          eval_p.gather_evaluate(src,
                                 EvaluationFlags::values |
                                   EvaluationFlags::gradients);

          eval_m.reinit(face);
          eval_m.gather_evaluate(src,
                                 EvaluationFlags::values |
                                   EvaluationFlags::gradients);

          const auto tau_ip =
            (std::abs((eval_m.get_normal_vector(0) *
                       eval_m.inverse_jacobian(0))[dim - 1]) +
             std::abs((eval_p.get_normal_vector(0) *
                       eval_p.inverse_jacobian(0))[dim - 1])) *
            Number(parameters.viscosity * (degree + 1) * (degree + 1));

          for (const unsigned int q : eval_m.quadrature_point_indices())
            {
              const auto w_m      = eval_m.get_value(q);
              const auto w_p      = eval_p.get_value(q);
              const auto normal   = eval_m.get_normal_vector(q);
              const auto grad_w_m = eval_m.get_gradient(q);
              const auto grad_w_p = eval_p.get_gradient(q);

              const auto flux_q1 = viscous_flux(w_m, grad_w_m, parameters);
              const auto flux_q2 = viscous_flux(w_p, grad_w_p, parameters);
              Tensor<1, dim, VectorizedArrayType> value_flux;
              for (unsigned int d = 0; d < dim; ++d)
                value_flux[d] = 0.5 * (flux_q1[d] * normal);
              for (unsigned int d = 0; d < dim; ++d)
                value_flux[d] += 0.5 * (flux_q2[d] * normal);
              for (unsigned int d = 0; d < dim; ++d)
                value_flux[d] -= tau_ip * (w_m[d + 1] - w_p[d + 1]);
              eval_vel_m.submit_value((-time_step * 0.5) * value_flux, q);
              eval_vel_p.submit_value((time_step * 0.5) * value_flux, q);

              Tensor<1, dim + 2, Tensor<1, dim, VectorizedArrayType>> w_jump;
              for (unsigned int d = 0; d < dim + 2; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                  w_jump[d][e] = (w_m[d] - w_p[d]) * (Number(0.5) * normal[e]);
              eval_vel_m.submit_gradient(
                (-time_step * 0.5) * viscous_flux(w_m, w_jump, parameters), q);
              eval_vel_p.submit_gradient(
                (-time_step * 0.5) * viscous_flux(w_p, w_jump, parameters), q);
            }

          eval_vel_m.integrate_scatter(EvaluationFlags::values |
                                         EvaluationFlags::gradients,
                                       dst);
          eval_vel_p.integrate_scatter(EvaluationFlags::values |
                                         EvaluationFlags::gradients,
                                       dst);
        }
    }

    void
    velocity_rhs_boundary(const MatrixFreeType &,
                          VectorType &,
                          const VectorType &,
                          const std::pair<unsigned int, unsigned int> &)
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    energy_rhs(const MatrixFreeType &                       data,
               VectorType &                                 dst,
               const VectorType &                           src,
               const std::pair<unsigned int, unsigned int> &cell_range)
    {
      FEEvaluation<dim, -1, 0, 1, Number>   eval_den(data, 0, 0, 0);
      FEEvaluation<dim, -1, 0, dim, Number> eval_euler_vel(data, 0, 0, 1);
      FEEvaluation<dim, -1, 0, 1, Number>   eval_ene(data, 0, 0, dim + 1);
      FEEvaluation<dim, -1, 0, dim, Number> eval_vel(data, 1);
      FEEvaluation<dim, -1, 0, 1, Number>   eval_energy(data, 2);
      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          eval_ene.reinit(cell);
          eval_den.reinit(cell);
          eval_euler_vel.reinit(cell);
          eval_vel.reinit(cell);
          eval_energy.reinit(cell);
          eval_ene.gather_evaluate(src, EvaluationFlags::values);
          eval_den.gather_evaluate(src, EvaluationFlags::values);
          eval_euler_vel.gather_evaluate(src, EvaluationFlags::values);
          eval_vel.gather_evaluate(velocity_solution,
                                   EvaluationFlags::gradients |
                                     EvaluationFlags::values);
          for (unsigned int q : eval_ene.quadrature_point_indices())
            {
              const auto                density  = eval_den.get_value(q);
              const auto                energy   = eval_ene.get_value(q);
              const auto                momentum = eval_euler_vel.get_value(q);
              const auto                grad_u   = eval_vel.get_gradient(q);
              const VectorizedArrayType scaled_div_u =
                parameters.viscosity * (2. / 3.) * trace(grad_u);
              Tensor<2, dim, VectorizedArrayType> viscous_stress;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  for (unsigned int e = d; e < dim; ++e)
                    {
                      viscous_stress[d][e] =
                        parameters.viscosity * (grad_u[d][e] + grad_u[e][d]);
                      viscous_stress[e][d] = viscous_stress[d][e];
                    }
                  viscous_stress[d][d] -= scaled_div_u;
                }
              eval_energy.submit_gradient(VectorizedArrayType(-time_step) *
                                            (viscous_stress *
                                             eval_vel.get_value(q)),
                                          q);
              eval_energy.submit_value(energy -
                                         0.5 * momentum.norm_square() / density,
                                       q);
            }

          eval_energy.integrate_scatter(EvaluationFlags::values |
                                          EvaluationFlags::gradients,
                                        dst);
        }
    }

    void
    update_velocity(
      const MatrixFreeType &                       data,
      VectorType &                                 dst,
      const VectorType &                           src,
      const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, dim + 2, Number> eval_euler(data, 0);
      FEEvaluation<dim, -1, 0, dim, Number>     eval_velocity(data, 1);
      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          eval_euler.reinit(cell);
          eval_velocity.reinit(cell);
          eval_euler.read_dof_values(dst);
          eval_velocity.read_dof_values(src);
          const unsigned int dofs_per_component =
            eval_velocity.dofs_per_component;
          for (unsigned int i = 0; i < dofs_per_component; ++i)
            {
              const auto density = eval_euler.begin_dof_values()[i];
              for (unsigned int d = 0; d < dim; ++d)
                {
                  eval_euler
                    .begin_dof_values()[i + (d + 1) * dofs_per_component] =
                    density * eval_velocity
                                .begin_dof_values()[i + d * dofs_per_component];
                }
            }
          eval_euler.set_dof_values(dst);
        }
    }
  };



  template <int dim>
  class FlowProblem
  {
  public:
    FlowProblem(const Parameters &parameters);

    void
    run();

  private:
    void
    make_grid(const unsigned int n_refinements);

    void
    make_dofs();

    void
    refine_grid(const unsigned int refine_cycle);

    void
    output_results(const unsigned int result_number);

    LinearAlgebra::distributed::Vector<Number> solution;

    ConditionalOStream pcout;

    Triangulation<dim> triangulation;

    FESystem<dim>   fe;
    MappingQ<dim>   mapping;
    DoFHandler<dim> dof_handler;

    FESystem<dim>             fe_velocity;
    DoFHandler<dim>           dof_handler_velocity;
    AffineConstraints<double> constraints_velocity;

    FE_Q<dim>                 fe_energy;
    DoFHandler<dim>           dof_handler_energy;
    AffineConstraints<double> constraints_energy;

    TimerOutput timer;

    EulerOperator<dim, fe_degree, n_q_points_1d> euler_operator;
    ViscousOperator<dim, fe_degree>              viscous_operator;

    const Parameters &parameters;

    double time, time_step;

    class Postprocessor : public DataPostprocessor<dim>
    {
    public:
      Postprocessor(const Parameters &parameters);

      virtual void
      evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;

      virtual std::vector<std::string>
      get_names() const override;

      virtual std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const override;

      virtual UpdateFlags
      get_needed_update_flags() const override;

    private:
      const bool        do_schlieren_plot;
      const Parameters &parameters;
    };
  };



  template <int dim>
  FlowProblem<dim>::FlowProblem(const Parameters &parameters)
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD)
#endif
    , fe(FE_DGQ<dim>(fe_degree), dim + 2)
    , mapping(fe_degree)
    , dof_handler(triangulation)
    , fe_velocity(FE_Q<dim>(fe_degree), dim)
    , dof_handler_velocity(triangulation)
    , fe_energy(fe_degree)
    , dof_handler_energy(triangulation)
    , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    , euler_operator(timer, parameters)
    , viscous_operator(euler_operator.get_matrix_free(), parameters)
    , parameters(parameters)
    , time(0)
    , time_step(0)
  {}



  template <int dim>
  void
  FlowProblem<dim>::make_grid(const unsigned int n_refinements)
  {
    TimerOutput::Scope t(timer, "setup grid");
    Point<dim>         lower_left, upper_right;
    for (unsigned int d = 0; d < dim; ++d)
      lower_left[d] = -numbers::PI;

    for (unsigned int d = 0; d < dim; ++d)
      upper_right[d] = numbers::PI;

    std::vector<unsigned int> refinements(dim, 1);
    for (unsigned int d = 0; d < std::min<unsigned int>(dim, n_refinements);
         ++d)
      refinements[d] = 2;
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              refinements,
                                              lower_left,
                                              upper_right);
    for (const auto &cell : triangulation.cell_iterators())
      for (unsigned int face : cell->face_indices())
        if (cell->at_boundary(face))
          cell->face(face)->set_boundary_id(face);
    std::vector<
      GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;
    for (unsigned int d = 0; d < dim; ++d)
      GridTools::collect_periodic_faces(
        triangulation, 2 * d, 2 * d + 1, d, periodic_faces);
    triangulation.add_periodicity(periodic_faces);

    triangulation.refine_global(2);
  }



  template <int dim>
  void
  FlowProblem<dim>::make_dofs()
  {
    TimerOutput::Scope t(timer, "setup dofs and operators");
    dof_handler.distribute_dofs(fe);
    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_energy.distribute_dofs(fe_energy);

    euler_operator.reinit(mapping,
                          dof_handler,
                          dof_handler_velocity,
                          constraints_velocity,
                          dof_handler_energy,
                          constraints_energy);
    euler_operator.initialize_vector(solution);
    viscous_operator.initialize();


    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale("en_US.UTF8"));
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << " ( = " << (dim + 2) << " [vars] x "
          << triangulation.n_global_active_cells() << " [cells] x "
          << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
          << std::endl;
    pcout.get_stream().imbue(s);
  }



  template <int dim>
  void
  FlowProblem<dim>::refine_grid(const unsigned int refine_cycle)
  {
    const unsigned int n_global_levels = 3;
    // To create a robust mesh, do not follow error indicators (the content in
    // high polynomial degrees would be one option), but instead just refine
    // into different radii
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const Point<dim> center = cell->center();
          if (center.norm() < (0.7 + 0.15 * (refine_cycle % 2)) * numbers::PI)
            {
              if (cell->level() < static_cast<int>(n_global_levels))
                cell->set_refine_flag();
            }
          else
            {
              if (cell->level() == static_cast<int>(n_global_levels))
                cell->set_coarsen_flag();
            }
        }

    const LinearAlgebra::distributed::Vector<Number> old_solution = solution;
    SolutionTransfer<dim, LinearAlgebra::distributed::Vector<Number>>
      solution_transfer(dof_handler);

    {
      TimerOutput::Scope t(timer, "refine step 1");

      triangulation.prepare_coarsening_and_refinement();

      solution_transfer.prepare_for_coarsening_and_refinement(old_solution);

      triangulation.execute_coarsening_and_refinement();
    }

    make_dofs();

    {
      TimerOutput::Scope t(timer, "refine step 2");
      solution_transfer.interpolate(old_solution, solution);
    }
  }



  template <int dim>
  FlowProblem<dim>::Postprocessor::Postprocessor(const Parameters &parameters)
    : do_schlieren_plot(dim == 2)
    , parameters(parameters)
  {}



  template <int dim>
  void
  FlowProblem<dim>::Postprocessor::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    const unsigned int n_evaluation_points = inputs.solution_values.size();

    if (do_schlieren_plot == true)
      Assert(inputs.solution_gradients.size() == n_evaluation_points,
             ExcInternalError());

    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == dim + 2, ExcInternalError());
    Assert(computed_quantities[0].size() ==
             dim + 2 + (do_schlieren_plot == true ? 1 : 0),
           ExcInternalError());

    for (unsigned int q = 0; q < n_evaluation_points; ++q)
      {
        Tensor<1, dim + 2> solution;
        for (unsigned int d = 0; d < dim + 2; ++d)
          solution[d] = inputs.solution_values[q](d);

        const double         density  = solution[0];
        const Tensor<1, dim> velocity = euler_velocity<dim>(solution);
        const double pressure = euler_pressure<dim>(solution, parameters);

        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[q](d) = velocity[d];
        computed_quantities[q](dim) = pressure;
        computed_quantities[q](dim + 1) =
          std::sqrt(parameters.gamma * pressure / density);

        if (do_schlieren_plot == true)
          computed_quantities[q](dim + 2) =
            inputs.solution_gradients[q][0] * inputs.solution_gradients[q][0];
      }
  }



  template <int dim>
  std::vector<std::string>
  FlowProblem<dim>::Postprocessor::get_names() const
  {
    std::vector<std::string> names;
    for (unsigned int d = 0; d < dim; ++d)
      names.emplace_back("velocity");
    names.emplace_back("pressure");
    names.emplace_back("speed_of_sound");

    if (do_schlieren_plot == true)
      names.emplace_back("schlieren_plot");

    return names;
  }



  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  FlowProblem<dim>::Postprocessor::get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation;
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    if (do_schlieren_plot == true)
      interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }



  template <int dim>
  UpdateFlags
  FlowProblem<dim>::Postprocessor::get_needed_update_flags() const
  {
    if (do_schlieren_plot == true)
      return update_values | update_gradients;
    else
      return update_values;
  }



  template <int dim>
  void
  FlowProblem<dim>::output_results(const unsigned int result_number)
  {
    TimerOutput::Scope          t(timer, "output");
    const std::array<double, 2> energy =
      euler_operator.compute_kinetic_energy(solution);
    pcout << "Time:" << std::setw(8) << std::setprecision(3) << time
          << ", dt: " << std::setw(8) << std::setprecision(2) << time_step
          << ", kinetic energy: " << std::setprecision(7) << std::setw(10)
          << energy[0] << ", dissipation: " << std::setprecision(4)
          << std::setw(10) << energy[1] << std::endl;

    Postprocessor postprocessor(parameters);
    DataOut<dim>  data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    {
      std::vector<std::string> names;
      names.emplace_back("density");
      for (unsigned int d = 0; d < dim; ++d)
        names.emplace_back("momentum");
      names.emplace_back("energy");

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation;
      interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
      for (unsigned int d = 0; d < dim; ++d)
        interpretation.push_back(
          DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

      data_out.add_data_vector(dof_handler, solution, names, interpretation);
    }
    data_out.add_data_vector(solution, postprocessor);
    constraints_velocity.distribute(viscous_operator.velocity_solution);
    data_out.add_data_vector(
      dof_handler_velocity,
      viscous_operator.velocity_solution,
      std::vector<std::string>(dim, "velocity_continuous"),
      std::vector<DataComponentInterpretation::DataComponentInterpretation>(
        dim, DataComponentInterpretation::component_is_part_of_vector));

    Vector<double> mpi_owner(triangulation.n_active_cells());
    mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector(mpi_owner, "owner");

    data_out.build_patches(mapping,
                           fe.degree,
                           DataOut<dim>::curved_inner_cells);
  }



  template <int dim>
  void
  FlowProblem<dim>::run()
  {
    pcout << "Running with T=" << parameters.end_time
          << " n_refine=" << parameters.n_refinements << std::endl;
    make_grid(parameters.n_refinements);

    make_dofs();

    euler_operator.project(InitialCondition<dim>(time, parameters), solution);
    refine_grid(0);

    double min_vertex_distance = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        min_vertex_distance =
          std::min(min_vertex_distance, cell->minimum_vertex_distance());
    min_vertex_distance =
      Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

    time_step = parameters.courant_number /
                euler_operator.compute_cell_transport_speed(solution);
    pcout << "Time step size: " << time_step
          << ", minimal h: " << min_vertex_distance
          << ", initial transport scaling: "
          << 1. / euler_operator.compute_cell_transport_speed(solution)
          << std::endl
          << std::endl;

    LinearAlgebra::distributed::Vector<Number> rk_register_1, rk_register_2;
    rk_register_1.reinit(solution);
    rk_register_2.reinit(solution);

    output_results(0);

    unsigned int timestep_number = 0;

    while (time < parameters.end_time - 1e-12)
      {
        ++timestep_number;
        if (timestep_number % 5 == 0)
          time_step =
            parameters.courant_number /
            Utilities::truncate_to_n_digits(
              euler_operator.compute_cell_transport_speed(solution), 3);

        {
          TimerOutput::Scope t(timer, "euler step");

          // propagate with Ralston's (sometimes called Heun's) method
          euler_operator.apply(time, solution, rk_register_1);
          solution.add(time_step * (0.5 * 1. / 4.), rk_register_1);
          rk_register_1.sadd(time_step * 0.5 * (2. / 3. - 1. / 4.),
                             1.,
                             solution);
          euler_operator.apply(time, rk_register_1, rk_register_2);
          solution.add(time_step * (0.5 * 3. / 4.), rk_register_2);
        }
        {
          TimerOutput::Scope t(timer, "viscous step");
          viscous_operator.propagate(time_step, solution);
        }
        {
          TimerOutput::Scope t(timer, "euler step");
          euler_operator.apply(time, solution, rk_register_1);
          solution.add(time_step * (0.5 * 1. / 4.), rk_register_1);
          rk_register_1.sadd(time_step * 0.5 * (2. / 3. - 1. / 4.),
                             1.,
                             solution);
          euler_operator.apply(time, rk_register_1, rk_register_2);
          solution.add(time_step * (0.5 * 3. / 4.), rk_register_2);
        }

        time += time_step;

        if (static_cast<int>(time / parameters.output_tick) !=
              static_cast<int>((time - time_step) / parameters.output_tick) ||
            time >= parameters.end_time - 1e-12)
          output_results(static_cast<unsigned int>(
            std::round(time / parameters.output_tick)));

        if (static_cast<int>(time / parameters.refine_tick) !=
            static_cast<int>((time - time_step) / parameters.refine_tick))
          {
            refine_grid(static_cast<unsigned int>(
              std::round(time / parameters.refine_tick)));
            rk_register_1.reinit(solution);
            rk_register_2.reinit(solution);
          }
      }

    viscous_operator.print_solver_statistics();
    pcout << std::endl;

    if (parameters.print_debug_timings)
      timer.print_wall_time_statistics(MPI_COMM_WORLD);
  }

} // namespace NavierStokes_DG

int
main(int argc, char **argv)
{
  using namespace NavierStokes_DG;
  using namespace dealii;

  try
    {
      MultithreadInfo::set_thread_limit(1);

      std::string filename = "parameters.prm";
      if (argc > 1)
        filename = argv[1];

      const Parameters       parameters(filename);
      FlowProblem<dimension> euler_problem(parameters);
      euler_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
