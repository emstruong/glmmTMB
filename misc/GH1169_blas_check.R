## integer-valued matrices: exact products are representable, so any error is a real BLAS bug
set.seed(1); bad <- 0
for (n in c(2,3,5,8,9,16,17,24,31,32,33,48,64,100,128,200,256,500)) for (k in c(1,7,8,23,64,300)) {
  A <- matrix(sample(-5:5, n*k, TRUE), n, k); B <- matrix(sample(-5:5, k*n, TRUE), k, n)
  ref <- matrix(0, n, n); for (j in 1:k) ref <- ref + outer(A[,j], B[j,])
  e1 <- max(abs(A %*% B - ref)); e2 <- max(abs(crossprod(t(A)) - tcrossprod(A) ))
  e3 <- max(abs(tcrossprod(A) - A %*% t(A)))
  if (e1 > 0 || e3 > 0) { bad <- bad+1; if (bad <= 8) cat(sprintf("n=%d k=%d  gemm err=%g  syrk-vs-gemm err=%g\n", n, k, e1, e3)) }
}
cat(Sys.getenv("TAG"), "bad cases:", bad, "\n")
## Cholesky: SPD integer-ish matrix
X <- matrix(sample(-3:3, 600*40, TRUE), 600, 40); M <- crossprod(X) + diag(40)
R <- chol(M); cat("chol resid:", max(abs(crossprod(R) - M)), "\n")
