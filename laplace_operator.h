
#ifndef poisson_operator_h
#define poisson_operator_h

#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>


namespace Laplace
{
  using namespace dealii;

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number
  do_invert(Tensor<2, 2, Number> &t)
  {
    const Number det     = t[0][0] * t[1][1] - t[1][0] * t[0][1];
    const Number inv_det = 1.0 / det;
    const Number tmp     = inv_det * t[0][0];
    t[0][0]              = inv_det * t[1][1];
    t[0][1]              = -inv_det * t[0][1];
    t[1][0]              = -inv_det * t[1][0];
    t[1][1]              = tmp;
    return det;
  }


  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number
  do_invert(Tensor<2, 3, Number> &t)
  {
    const Number tr00    = t[1][1] * t[2][2] - t[1][2] * t[2][1];
    const Number tr10    = t[1][2] * t[2][0] - t[1][0] * t[2][2];
    const Number tr20    = t[1][0] * t[2][1] - t[1][1] * t[2][0];
    const Number det     = t[0][0] * tr00 + t[0][1] * tr10 + t[0][2] * tr20;
    const Number inv_det = 1.0 / det;
    const Number tr01    = t[0][2] * t[2][1] - t[0][1] * t[2][2];
    const Number tr02    = t[0][1] * t[1][2] - t[0][2] * t[1][1];
    const Number tr11    = t[0][0] * t[2][2] - t[0][2] * t[2][0];
    const Number tr12    = t[0][2] * t[1][0] - t[0][0] * t[1][2];
    t[2][1]              = inv_det * (t[0][1] * t[2][0] - t[0][0] * t[2][1]);
    t[2][2]              = inv_det * (t[0][0] * t[1][1] - t[0][1] * t[1][0]);
    t[0][0]              = inv_det * tr00;
    t[0][1]              = inv_det * tr01;
    t[0][2]              = inv_det * tr02;
    t[1][0]              = inv_det * tr10;
    t[1][1]              = inv_det * tr11;
    t[1][2]              = inv_det * tr12;
    t[2][0]              = inv_det * tr20;
    return det;
  }



  namespace internal
  {
    template <typename Number>
    Number
    set_constrained_entries(
      const std::vector<unsigned int> &                 constrained_entries,
      const LinearAlgebra::distributed::Vector<Number> &src,
      LinearAlgebra::distributed::Vector<Number> &      dst)
    {
      Number sum = 0;
      for (unsigned int i = 0; i < constrained_entries.size(); ++i)
        {
          dst.local_element(constrained_entries[i]) =
            src.local_element(constrained_entries[i]);
          sum += src.local_element(constrained_entries[i]) *
                 src.local_element(constrained_entries[i]);
        }
      return sum;
    }
  } // namespace internal



  template <int dim, int n_components, typename Number>
  struct KernelBase
  {
    virtual void
    pointwise_apply(Tensor<1, n_components, Number> &                value,
                    Tensor<1, n_components, Tensor<1, dim, Number>> &grad,
                    const unsigned int                               cell,
                    const unsigned int q_index) const = 0;
  };


  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_    = 1,
            typename Number      = double,
            typename VectorType_ = LinearAlgebra::distributed::Vector<Number>,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class LaplaceOperator : public Subscriptor
  {
  public:
    /**
     * Number typedef.
     */
    using value_type = Number;

    /**
     * size_type needed for preconditioner classes.
     */
    using size_type = types::global_dof_index;

    /**
     * size_type needed for preconditioner classes.
     */
    using VectorType = VectorType_;

    /**
     * Make number of components available as variable
     */
    static constexpr unsigned int n_components = n_components_;

    /**
     * Constructor.
     */
    LaplaceOperator(
      const KernelBase<dim, n_components, VectorizedArrayType> &kernel)
      : dof_index(0)
      , quad_index(0)
      , kernel(kernel)
    {}

    /**
     * Initialize function.
     */
    void
    initialize(const MatrixFree<dim, Number, VectorizedArrayType> &data_,
               const unsigned int                                  dof_index,
               const unsigned int                                  quad_index)
    {
      compute_time                                 = 0.;
      const_cast<unsigned int &>(this->dof_index)  = dof_index;
      const_cast<unsigned int &>(this->quad_index) = quad_index;
      this->data                                   = &data_;
      quad_1d = data->get_shape_info(dof_index, quad_index).data[0].quadrature;
      cell_quadratic_coefficients.resize(data->n_cell_batches());

      FE_Nothing<dim> dummy_fe;
      FEValues<dim>   fe_values(dummy_fe,
                              QGaussLobatto<dim>(3),
                              update_quadrature_points);

      for (unsigned int c = 0; c < data->n_cell_batches(); ++c)
        {
          for (unsigned int l = 0; l < data->n_active_entries_per_cell_batch(c);
               ++l)
            {
              const typename Triangulation<dim>::cell_iterator cell =
                data->get_cell_iterator(c, l, dof_index);
              fe_values.reinit(cell);
              const double coeff[9] = {
                1.0, -3.0, 2.0, 0.0, 4.0, -4.0, 0.0, -1.0, 2.0};
              constexpr unsigned int size_dim = Utilities::pow(3, dim);
              std::array<Tensor<1, dim>, size_dim> points;
              for (unsigned int i2 = 0; i2 < (dim > 2 ? 3 : 1); ++i2)
                {
                  for (unsigned int i1 = 0; i1 < 3; ++i1)
                    for (unsigned int i0 = 0, i = 9 * i2 + 3 * i1; i0 < 3; ++i0)
                      points[i + i0] =
                        coeff[i0] * fe_values.quadrature_point(i) +
                        coeff[i0 + 3] * fe_values.quadrature_point(i + 1) +
                        coeff[i0 + 6] * fe_values.quadrature_point(i + 2);
                  for (unsigned int i1 = 0; i1 < 3; ++i1)
                    {
                      const unsigned int            i   = 9 * i2 + i1;
                      std::array<Tensor<1, dim>, 3> tmp = {
                        {points[i], points[i + 3], points[i + 6]}};
                      for (unsigned int i0 = 0; i0 < 3; ++i0)
                        points[i + 3 * i0] = coeff[i0] * tmp[0] +
                                             coeff[i0 + 3] * tmp[1] +
                                             coeff[i0 + 6] * tmp[2];
                    }
                }
              if (dim == 3)
                for (unsigned int i = 0; i < 9; ++i)
                  {
                    std::array<Tensor<1, dim>, 3> tmp = {
                      {points[i], points[i + 9], points[i + 18]}};
                    for (unsigned int i0 = 0; i0 < 3; ++i0)
                      points[i + 9 * i0] = coeff[i0] * tmp[0] +
                                           coeff[i0 + 3] * tmp[1] +
                                           coeff[i0 + 6] * tmp[2];
                  }
              for (unsigned int i = 0; i < points.size(); ++i)
                for (unsigned int d = 0; d < dim; ++d)
                  cell_quadratic_coefficients[c][i][d][l] = points[i][d];
            }
        }
    }

    void
    clear()
    {
      data = nullptr;
      cell_quadratic_coefficients.clear();
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      this->data->cell_loop(
        &LaplaceOperator::local_apply_quadratic_geo, this, dst, src, true);
      internal::set_constrained_entries(data->get_constrained_dofs(0),
                                        src,
                                        dst);
    }

    void
    vmult(VectorType &      dst,
          const VectorType &src,
          const std::function<void(const unsigned int, const unsigned int)>
            &operation_before_loop,
          const std::function<void(const unsigned int, const unsigned int)>
            &operation_after_loop) const
    {
      Timer time;
      this->data->cell_loop(&LaplaceOperator::local_apply_quadratic_geo,
                            this,
                            dst,
                            src,
                            operation_before_loop,
                            operation_after_loop,
                            dof_index);
      compute_time += time.wall_time();
    }

    double
    get_compute_time_and_reset()
    {
      const auto   comm   = data->get_dof_handler().get_communicator();
      const double result = Utilities::MPI::sum(compute_time, comm) /
                            Utilities::MPI::n_mpi_processes(comm);
      compute_time = 0.;
      return result;
    }

    VectorType
    compute_mass_diagonal(const VectorType &euler_solution) const
    {
      VectorType help, diag;
      data->initialize_dof_vector(diag, dof_index);
      help.reinit(diag, true);
      help = 1.;
      data->template cell_loop<VectorType, VectorType>(
        [&](const MatrixFree<dim, value_type, VectorizedArrayType> &data,
            VectorType &                                            dst,
            const VectorType &                                      src,
            const std::pair<unsigned int, unsigned int> &cell_range) {
          FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_density(
            data, 0, quad_index);
          FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number>
            eval(data, dof_index, quad_index);
          for (unsigned int cell = cell_range.first; cell < cell_range.second;
               ++cell)
            {
              eval.reinit(cell);
              eval_density.reinit(cell);
              eval_density.gather_evaluate(euler_solution,
                                           EvaluationFlags::values);
              eval.gather_evaluate(src, EvaluationFlags::values);
              for (unsigned int q : eval.quadrature_point_indices())
                eval.submit_value(eval_density.get_value(q) * eval.get_value(q),
                                  q);
              eval.integrate_scatter(EvaluationFlags::values, dst);
            }
        },
        diag,
        help);
      return diag;
    }

    VectorType
    compute_stiff_diagonal() const
    {
      VectorType diag;
      data->initialize_dof_vector(diag, dof_index);

      MatrixFreeTools::
        compute_diagonal<dim, -1, 0, n_components, Number, VectorizedArrayType>(
          *data,
          diag,
          [&](auto &eval) {
            eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            for (unsigned int q : eval.quadrature_point_indices())
              {
                Tensor<1, n_components, VectorizedArrayType> val =
                  eval.get_value(q);
                Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
                  grad = eval.get_gradient(q);
                kernel.pointwise_apply(val,
                                       grad,
                                       eval.get_current_cell_index(),
                                       q);
                eval.submit_gradient(grad, q);
              }
            eval.integrate(EvaluationFlags::gradients);
          },
          dof_index,
          quad_index);
      return diag;
    }

  private:
    void
    local_apply_quadratic_geo(
      const MatrixFree<dim, value_type, VectorizedArrayType> &data,
      VectorType &                                            dst,
      const VectorType &                                      src,
      const std::pair<unsigned int, unsigned int> &           cell_range) const
    {
      FEEvaluation<dim,
                   fe_degree,
                   n_q_points_1d,
                   n_components,
                   Number,
                   VectorizedArrayType>
                             phi(data, dof_index, quad_index);
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
      AssertDimension(n_q_points, phi.n_q_points);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      std::array<TensorType, Utilities::pow(3, dim - 1)> xi;
      std::array<TensorType, Utilities::pow(3, dim - 1)> di;
      using Eval = dealii::internal::EvaluatorTensorProduct<
        dealii::internal::evaluate_evenodd,
        dim,
        n_q_points_1d,
        n_q_points_1d,
        VectorizedArrayType,
        VectorizedArrayType>;

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          phi.evaluate(EvaluationFlags::values);

          const auto &               v = cell_quadratic_coefficients[cell];
          VectorizedArrayType *      phi_grads = phi.begin_gradients();
          const VectorizedArrayType *shape_grads =
            phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin();
          if (dim == 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<1, true, false, 1>(
                    shape_grads,
                    phi.begin_values() + c * n_q_points,
                    phi_grads + (2 * c + 1) * n_q_points);
                  Eval::template apply<0, true, false, 1>(shape_grads,
                                                          phi.begin_values() +
                                                            c * n_q_points,
                                                          phi_grads +
                                                            2 * c * n_q_points);
                }
              for (unsigned int q = 0, qy = 0; qy < n_q_points_1d; ++qy)
                {
                  const Number     y  = quad_1d.point(qy)[0];
                  const TensorType x1 = v[1] + y * (v[4] + y * v[7]);
                  const TensorType x2 = v[2] + y * (v[5] + y * v[8]);
                  const TensorType d0 = v[3] + (y + y) * v[6];
                  const TensorType d1 = v[4] + (y + y) * v[7];
                  const TensorType d2 = v[5] + (y + y) * v[8];
                  for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                    {
                      const Number q_weight =
                        quad_1d.weight(qy) * quad_1d.weight(qx);
                      const Number x = quad_1d.point(qx)[0];
                      Tensor<2, dim, VectorizedArrayType> jac;
                      jac[0]                  = x1 + (x + x) * x2;
                      jac[1]                  = d0 + x * d1 + (x * x) * d2;
                      VectorizedArrayType det = do_invert(jac);
                      det                     = det * q_weight;

                      Tensor<1,
                             n_components,
                             Tensor<1, dim, VectorizedArrayType>>
                                                                   grad;
                      Tensor<1, n_components, VectorizedArrayType> val;
                      for (unsigned int c = 0; c < n_components; ++c)
                        {
                          val[c] = phi.begin_values()[c * n_q_points + q];
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              grad[c][d] =
                                jac[d][0] * phi_grads[q + c * 2 * n_q_points] +
                                jac[d][1] *
                                  phi_grads[q + (c * 2 + 1) * n_q_points];
                            }
                        }
                      kernel.pointwise_apply(val, grad, cell, q);
                      for (unsigned int c = 0; c < n_components; ++c)
                        {
                          phi.begin_values()[c * n_q_points + q] = val[c] * det;
                          VectorizedArrayType tmp2[dim];
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp2[d] = jac[0][d] * grad[c][0];
                              for (unsigned int e = 1; e < dim; ++e)
                                tmp2[d] += jac[e][d] * grad[c][e];
                              tmp2[d] *= det;
                            }
                          phi_grads[q + c * 2 * n_q_points]       = tmp2[0];
                          phi_grads[q + (c * 2 + 1) * n_q_points] = tmp2[1];
                        }
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<0, false, true, 1>(shape_grads,
                                                          phi_grads +
                                                            2 * c * n_q_points,
                                                          phi.begin_values() +
                                                            c * n_q_points);
                  Eval::template apply<1, false, true, 1>(
                    shape_grads,
                    phi_grads + (2 * c + 1) * n_q_points,
                    phi.begin_values() + c * n_q_points);
                }
            }
          else if (dim == 3)
            {
              constexpr unsigned int n_q_points_2d =
                n_q_points_1d * n_q_points_1d;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<2, true, false, 1>(
                    shape_grads,
                    phi.begin_values() + c * n_q_points,
                    phi_grads + (2 * n_components * n_q_points_2d) +
                      c * n_q_points);
                }
              for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
                {
                  using Eval2 = dealii::internal::EvaluatorTensorProduct<
                    dealii::internal::evaluate_evenodd,
                    2,
                    n_q_points_1d,
                    n_q_points_1d,
                    VectorizedArrayType,
                    VectorizedArrayType>;
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<1, true, false, 1>(
                        shape_grads,
                        phi.begin_values() + c * n_q_points +
                          qz * n_q_points_2d,
                        phi_grads + (2 * c + 1) * n_q_points_2d);
                      Eval2::template apply<0, true, false, 1>(
                        shape_grads,
                        phi.begin_values() + c * n_q_points +
                          qz * n_q_points_2d,
                        phi_grads + 2 * c * n_q_points_2d);
                    }
                  const Number z = quad_1d.point(qz)[0];
                  di[0]          = v[9] + (z + z) * v[18];
                  for (unsigned int i = 1; i < 9; ++i)
                    {
                      xi[i] = v[i] + z * (v[9 + i] + z * v[18 + i]);
                      di[i] = v[9 + i] + (z + z) * v[18 + i];
                    }
                  for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                    {
                      const auto       y   = quad_1d.point(qy)[0];
                      const TensorType x1  = xi[1] + y * (xi[4] + y * xi[7]);
                      const TensorType x2  = xi[2] + y * (xi[5] + y * xi[8]);
                      const TensorType dy0 = xi[3] + (y + y) * xi[6];
                      const TensorType dy1 = xi[4] + (y + y) * xi[7];
                      const TensorType dy2 = xi[5] + (y + y) * xi[8];
                      const TensorType dz0 = di[0] + y * (di[3] + y * di[6]);
                      const TensorType dz1 = di[1] + y * (di[4] + y * di[7]);
                      const TensorType dz2 = di[2] + y * (di[5] + y * di[8]);
                      double           q_weight_tmp =
                        quad_1d.weight(qz) * quad_1d.weight(qy);
                      for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                        {
                          const Number x = quad_1d.point(qx)[0];
                          Tensor<2, dim, VectorizedArrayType> jac;
                          jac[0]                  = x1 + (x + x) * x2;
                          jac[1]                  = dy0 + x * (dy1 + x * dy2);
                          jac[2]                  = dz0 + x * (dz1 + x * dz2);
                          VectorizedArrayType det = do_invert(jac);
                          det = det * (q_weight_tmp * quad_1d.weight(qx));

                          Tensor<1,
                                 n_components,
                                 Tensor<1, dim, VectorizedArrayType>>
                                                                       grad;
                          Tensor<1, n_components, VectorizedArrayType> val;
                          for (unsigned int c = 0; c < n_components; ++c)
                            {
                              val[c] = phi.begin_values()[c * n_q_points + q];
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  grad[c][d] =
                                    jac[d][0] *
                                      phi_grads[qy * n_q_points_1d + qx +
                                                c * 2 * n_q_points_2d] +
                                    jac[d][1] *
                                      phi_grads[qy * n_q_points_1d + qx +
                                                (c * 2 + 1) * n_q_points_2d] +
                                    jac[d][2] * phi_grads[q +
                                                          2 * n_components *
                                                            n_q_points_2d +
                                                          c * n_q_points];
                                }
                            }
                          kernel.pointwise_apply(val, grad, cell, q);
                          for (unsigned int c = 0; c < n_components; ++c)
                            {
                              phi.begin_values()[c * n_q_points + q] =
                                val[c] * det;
                              VectorizedArrayType tmp2[dim];
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  tmp2[d] = jac[0][d] * grad[c][0];
                                  for (unsigned int e = 1; e < dim; ++e)
                                    tmp2[d] += jac[e][d] * grad[c][e];
                                  tmp2[d] *= det;
                                }
                              phi_grads[qy * n_q_points_1d + qx +
                                        c * 2 * n_q_points_2d]       = tmp2[0];
                              phi_grads[qy * n_q_points_1d + qx +
                                        (c * 2 + 1) * n_q_points_2d] = tmp2[1];
                              phi_grads[q + 2 * n_components * n_q_points_2d +
                                        c * n_q_points]              = tmp2[2];
                            }
                        }
                    }
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<0, false, true, 1>(
                        shape_grads,
                        phi_grads + 2 * c * n_q_points_2d,
                        phi.begin_values() + c * n_q_points +
                          qz * n_q_points_2d);
                      Eval2::template apply<1, false, true, 1>(
                        shape_grads,
                        phi_grads + (2 * c + 1) * n_q_points_2d,
                        phi.begin_values() + c * n_q_points +
                          qz * n_q_points_2d);
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<2, false, true, 1>(
                    shape_grads,
                    phi_grads + (2 * n_components * n_q_points_2d) +
                      c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                }
            }

          phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    const MatrixFree<dim, Number, VectorizedArrayType> *data;

    mutable double compute_time;

    Quadrature<1> quad_1d;

    // For local_apply_quadratic_geo
    AlignedVector<
      std::array<Tensor<1, dim, VectorizedArrayType>, Utilities::pow(3, dim)>>
      cell_quadratic_coefficients;

    unsigned int dof_index;
    unsigned int quad_index;

    const KernelBase<dim, n_components, VectorizedArrayType> &kernel;
  };

} // namespace Laplace

#endif
