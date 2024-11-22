#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "lapack.h"

void polynomial_fit(int degree, const char* data_file, const char* output_file) {
    // Open the data file
    FILE* file = fopen(data_file, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read the number of data points
    int n;
    fscanf(file, "%d", &n);

    // Allocate memory for data points
    double* x = malloc(n * sizeof(double));
    double* y = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        fscanf(file, "%lf %lf", &x[i], &y[i]);
    }
    fclose(file);

    // Allocate memory for matrices
    int d = degree + 1;
    double* X = calloc(n * d, sizeof(double)); // Matrix X
    double* XTX = calloc(d * d, sizeof(double)); // Matrix X^T X
    double* XTy = calloc(d, sizeof(double)); // Vector X^T y
    double* coeff = calloc(d, sizeof(double)); // Coefficients

    // Fill matrix X
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            X[i * d + j] = pow(x[i], j);  // x[i] raised to the power j
            printf("%lf ", X[i * d + j]);  // Debugging: print each element
        }
        printf("\n");
    }

    // Compute X^T X using dgemm
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        d, d, n,
        1.0, X, d, X, d,
        0.0, XTX, d);

    // Compute X^T y using dgemv
    cblas_dgemv(CblasRowMajor, CblasTrans,
        n, d,
        1.0, X, d, y, 1,
        0.0, XTy, 1);



    // Perform Cholesky factorization on XTX
    int info = 0;  // Declare info as an integer, not a pointer
    dpotrf_("L", &d, XTX, &d, &info);  // Pass pointers where needed
    if (info == 0) {
        printf("Cholesky factorization successful.\n");
        printf("Lower triangular matrix L:\n");
        // Print the matrix L (lower triangular matrix)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i >= j) {
                    printf("%lf ", XTX[i * n + j]);
                }
                else {
                    printf("0.0 ");  // Fill upper triangular part with zeros
                }
            }
            printf("\n");
        }
    }
    else {
        printf("Error: Cholesky factorization failed.\n");
    }

    // Solve Lb = X^T y using dtrsm
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
        d, 1,
        1.0, XTX, d, XTy, 1);

    // Solve L^T c = b using dtrsm
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
        d, 1,
        1.0, XTX, d, XTy, 1);

    // Save the coefficients
    for (int i = 0; i < d; i++) {
        coeff[i] = XTy[i];
    }

    // Write results to output file
    FILE* output = fopen(output_file, "w");
    if (!output) {
        perror("Error opening output file");
        free(x);
        free(y);
        free(X);
        free(XTX);
        free(XTy);
        free(coeff);
        exit(EXIT_FAILURE);
    }

    fprintf(output, "# Original Data Points\n");
    for (int i = 0; i < n; i++) {
        fprintf(output, "%f %f\n", x[i], y[i]);
    }

    fprintf(output, "\n# x y y_fit\n");
    for (int i = 0; i < n; i++) {
        double y_fit = 0.0;
        for (int j = 0; j < d; j++) {
            y_fit += coeff[j] * pow(x[i], j);
        }
        fprintf(output, "%lf %lf %lf\n", x[i], y[i], y_fit);
    }
    fclose(output);

    // Free allocated memory
    free(x);
    free(y);
    free(X);
    free(XTX);
    free(XTy);
    free(coeff);

    printf("Polynomial fit written to %s\n", output_file);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <degree> <data_file> <output_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int degree = atoi(argv[1]);
    const char* data_file = argv[2];
    const char* output_file = argv[3];

    polynomial_fit(degree, data_file, output_file);
    return EXIT_SUCCESS;
}
