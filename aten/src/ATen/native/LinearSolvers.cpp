#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

namespace at {
namespace native {
namespace {
void checkSameDtype(const Tensor& t1,
                    const Tensor& t2,
                    const char* const f_name,
                    const char* const t1_name,
                    const char* const t2_name) {
  TORCH_CHECK(t1.scalar_type() == t2.scalar_type(),
              f_name, ": Expected ", t1_name, " and ", t2_name, " to have the same dtype. ",
              "Got ", t1_name, ".dtype = ", t1.scalar_type(),
              " and ", t2_name, ".dtype = ", t2.scalar_type());
}

void checkIsMatrix(const Tensor& t,
                   const char* const f_name,
                   const char* const t_name) {
  TORCH_CHECK(t.dim() >= 2, f_name, ": Expected ", t_name,
                            " to be a tensor of at least 2 dimensions.");
}

void checkIsSquareMatrix(const Tensor& t,
                         const char* const f_name,
                         const char* const t_name) {
  checkIsMatrix(t, f_name, t_name);
  TORCH_CHECK(self.size(-1) == self.size(-2),
              f_name, ": Expected ", t_name,
              " to be a square matrix or batch of square matrices. "
              "Got matrices of size (", t.size(-2), ", ", t.size(-1), ").");
}

void checkInputsSolver(const Tensor& A,
                       const Tensor& B,
                       const Tensor& out,
                       const char* const f_name) {
  checkSameDtype(A, B, f_name, "A", "B");
  checkSameDtype(A, out, f_name, "A", "out");
  checkIsSquareMatrix(A, f_name, "A");
  checkIsMatrix(B, f_name, "B");
  TORCH_CHECK(A.size(-1) == B.size(-2),
              name, ": Incompatible shapes. Each matrix in A is of shape (",
              A.size(-2), ", ", A.size(-1),
              ") but each matrix in B is of shape (",
              B.size(-2), ", ", B.size(-1), ")");
}

bool is_fortran_contiguous(const Tensor& T) {
  return T.transpose(-2, -1).is_contiguous();
}

} // end of anonymous namespace

DEFINE_DISPATCH(solve_triangular_stub);

/*
Solves the matrix equation AX = B for A triangular.
'upper' controls the portion of input matrix to consider in computations,
'unitriangular' if true then we assume diag(A) to be ones
'left' If true solves AX = B, if false solves XA = B
'out' The tensor with the result. If A == out, A will be modified in place
Returns:
The X that solves the linear system
'infos' which stores possible errors in the computation
*/
static std::tuple<Tensor, Tensor> solve_triangular_ex_out(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool unitriangular,
    bool left,
    bool check_errors,
    Tensor& out) {
  if (check_errors) {
    checkInputsSolver(A, B, out, "solve_triangular");
  }
  Tensor A_broad, B_broad;
  std::tie(B_broad, A_broad) = _linalg_broadcast_batch_dims(B, A, "solve_triangular", /*check_errors*/ false);
  Tensor infos = at::zeros({std::max<int64_t>(1, batchCount(A_broad))}, A.options().dtype(kInt));
  at::native::resize_output(out, result_tmp.sizes());

  // At this point, A, B have been broadcasted, infos is correct.
  // out may still be empty
  //
  // On CPU, if A is contiguous or FORTRAN-contig it is possible to avoid copying A in every case but
  // when left=True and A is FORTRAN-contig and conjugated and B has :
  //
  // 1) If left == False, we transpose A & B and we will transpose the solution.
  // 2) If A has the conj bit = true and is not transposed:
  //    Create a (FORTRAN-contig) conjugate copy of B and return the conjugate view of the solution. Take this chance of doing it in out directly.
  // 3) Set the bit to conjugate_transpose / trans / NoTrans as expected by lapack / magma / etc
  // 4) At this stage, if the following miracle happens:
  //    - B has not been copied yet
  //    - B is FORTRAN-contig
  //    - B is not conjugated and
  //    - B = out
  //    we don't copy

  // On CUDA, as it does implement natively left/right solvers, we can always avoid all
  // the copies if B == out and A & B are either contiguous or fortran contiguous
  //
  // 1) Make B fortran contiguous and not conj (and remember "applying"* the opposite operations to the returned tensor) *See the Remark below
  //    This gives a system of the form op1(A)op2(X) = B or op2(X)op1(A) = B where
  //    opi in {Normal, Conj, Trans, ConjTrans}
  //    We don't care about op2, as that's operations that we will perform before returning the solution
  //    We know how to solve all cases but op1 = Conj.
  // 2) If op1 = Conj, we set left = !left and op1 = ConjTrans.
  //
  // Remark: There is a subtle point here. Note that following the algorithm above, X has to
  //         be conjugated iff B was conjugated. This is particularly neat, as it is not possible
  //         to "unset" the conj bit in PyTorch---doing a.conj().conj() incurs in one copy.

  // To sumarise:
  // If B == out and the results are in CUDA, we can do a solver without copying memory
  // Otherwise, we can do it with at most one copy
  if (A.device() == at::kCPU && (A_broad.is_contiguous() || is_fortran_contiguous(A_broad))) {
    if (!left) {
      A_broad.transpose_(-2 ,-1);
      B_broad.transpose_(-2 ,-1);
    }
    // Here A may be F-contig / F-conj / F-transposed / F-conj-transposed
    // The only one that LAPACK doesn't know how to handle is F-conj

    // A is F-conj and not Fortran-transposed
    const bool A_is_conj = A_broad.is_conj() && is_fortran_contiguous(A_broad);
    if (A.is_conj()) {
      if (is_fotran_contiguous(A_broad)) {
        // CONTINUE HERE
      }
    }

    if (!left) {
      out.transpose_(-2, -1);
    }
    if (A_is_conj) {
      out.conj_();
    }
  } else { // CUDA
  }

  if (check_errors) {
    if (infos.dim() > 1) {
      batchCheckErrors(infos, "solve_triangular");
    } else {
      singleCheckErrors(infos.item().toInt(), "solve_triangular");
    }
  }
  return std::make_pair(out, infos);
}

static std::tuple<Tensor, Tensor> solve_triangular_ex(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool unitriangular,
    bool left,
    bool check_errors) {
  Tensor out = at::empty({0}, A.options());
  return solve_triangular_ex_out(A, B, upper, unitriangular, left, check_errors, out);
}

static Tensor solve_triangular_out(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool unitriangular,
    bool left,
    Tensor& out) {
  return std::get<0>(solve_triangular_ex_out(A, B, upper, unitriangular, left, /*check_errors*/true, out));
}

static Tensor solve_triangular(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool unitriangular,
    bool left) {
  Tensor out = at::empty({0}, A.options());
  return solve_triangular_out(A, B, upper, unitriangular, left, out);
}

}}  // namespace at::native
