#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "lapack.h"

void polynomial_fit(int degree, const char* data_file, const char* output_file, const char* L_file) {
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
            X[i * d + j] = pow(x[i], j);
        }
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

        // Save the lower triangular matrix L to L_file
        FILE* L_output = fopen(L_file, "w");
        if (!L_output) {
            perror("Error opening L_file");
            exit(EXIT_FAILURE);
        }

        // Print the matrix L (lower triangular matrix)
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (i >= j) {
                    fprintf(L_output, "%lf ", XTX[i * d + j]);
                } else {
                    fprintf(L_output, "0.0 ");  // Fill upper triangular part with zeros
                }
            }
            fprintf(L_output, "\n");
        }
        fclose(L_output);

    } else {
        printf("Error: Cholesky factorization failed.\n");
        exit(EXIT_FAILURE);
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
    fprintf(output, "# x y y_fit\n");
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

int main() {
    const char* data_file = "data.dat";  // Data file with x, y values

    // Loop over degrees 3 to 9
    for (int degree = 1; degree <= 9; degree++) {
        // Output file for polynomial fit
        char output_file[100];
        snprintf(output_file, sizeof(output_file), "output_degree%d.txt", degree);

        // Output file for lower triangular matrix L
        char L_file[100];
        snprintf(L_file, sizeof(L_file), "L_degree%d.txt", degree);

        // Perform polynomial fit and save results
        polynomial_fit(degree, data_file, output_file, L_file);
    }

    return EXIT_SUCCESS;
}
