void dgemv_(char* ta, int* m, int* n, double* alpha, double* a, int* lda,
            double *x, int* incx, double* beta, double* y, int* incy);
void dgemm_(char* ta, char* tb, int *m, int *n, int *k,
  double *alpha, double *a, int *lda, double *b, int *ldb,
  double *beta, double *c, int *ldc);

void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);
void dtrsm_(char* side, char* uplo, char* transa, char* diag,
            int* m, int* n, double* alpha, double* a, int* lda, 
            double* b, int* ldb);
