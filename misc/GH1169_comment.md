I think I've found the cause, and it isn't glmmTMB: on the problematic Intel machine, **the system BLAS (OpenBLAS) returns wrong results**, and `REML = TRUE` happens to be the code path in glmmTMB that runs into it.

**Setup:** Intel Xeon w7-2495X (Sapphire Rapids), Ubuntu 22.04.5, R 4.6.2, glmmTMB 1.1.15.2, TMB 1.9.25, with Ubuntu's stock OpenBLAS `0.3.20+ds-1` (`libopenblas0-pthread`). This OpenBLAS version auto-detects the CPU as **Cooperlake** (check with `OPENBLAS_VERBOSE=2`) and uses those kernels.

### OpenBLAS returns wrong results on this machine

The script below multiplies integer-valued matrices, so the exact products can be represented in floating point and any nonzero error is a real bug, not rounding:

```r
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
```

Run it under different OpenBLAS kernels, e.g. `OPENBLAS_CORETYPE=Haswell Rscript misc/GH1169_blas_check.R`:

| OpenBLAS kernel | wrong integer products | `chol()` residual |
|---|---|---|
| default (auto-detected Cooperlake) | 5 cases, errors of ~150–1200 | **181** |
| default, `OPENBLAS_NUM_THREADS=1` | 10 cases, errors of ~150–3600 | **181** |
| `OPENBLAS_CORETYPE=SkylakeX` | 0 | 4.5e-13 |
| `OPENBLAS_CORETYPE=Haswell` | 0 | 4.5e-13 |

### Effect on glmmTMB

With the kernel forced to SkylakeX, Haswell, Sandybridge or Prescott, everything matches the AMD machine / your results:

| | default (Cooperlake) | `OPENBLAS_CORETYPE=SkylakeX` / `Haswell` |
|---|---|---|
| `count ~ spp + (1\|site), zi = ~spp + mined, nbinom2, REML = TRUE` | false convergence (8), objective 830.632, ZI `minedno` = -14.2 | converges, objective 830.614, ZI `minedno` = -21.49 |
| `Reaction ~ Days + (Days\|Subject)`, REML, vs `lmer` | logLik differs by 0.155, vcov differs by up to 4.006 | logLik matches to 1e-10 |

The sleepstudy row reproduces the earlier `test-reml.R:16` failure exactly ("average diff: 4.006").

### Why only REML, and why only with non-intercept fixed effects

With `REML = TRUE`, glmmTMB adds `beta` to TMB's `random` set, so the fixed effects are Laplace-integrated together with `b`. The inner Hessian then has a dense `X'WX` block linked to every random effect, and factoring it with the sparse Cholesky makes dense BLAS/LAPACK calls. Under ML with `(1|site)` the inner Hessian is essentially diagonal and those calls aren't needed. With an intercept-only conditional model, the dense part is a single column, apparently too small to reach the broken kernel. That matches the earlier observation that any fixed effect beyond the intercept triggers the problem. (I haven't traced exactly which CHOLMOD/BLAS routine is involved; switching the kernel turns every symptom on and off.)

### Workarounds

- Set `OPENBLAS_CORETYPE=SkylakeX` (e.g. in `~/.Renviron`), or
- upgrade to a newer OpenBLAS with proper Sapphire Rapids support, or
- switch to a different BLAS (reference BLAS or MKL) via `update-alternatives`.

Any of these can be checked by re-running the script above, which should report `bad cases: 0` and a `chol` residual around 1e-13.

Since this affects anything in R that calls BLAS (including `lme4` once problems are big enough), I think it's worth a mention in the troubleshooting vignette: on new Intel server chips with an older distro OpenBLAS, running a quick check like the one above before trusting results would have caught this right away. I'm happy to close the issue if you agree it's not a glmmTMB problem.
