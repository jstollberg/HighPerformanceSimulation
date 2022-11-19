package task1;

import java.util.Random;

public class MatrixVector {

    int m;
    double[][] matrix;
    double[] vector;

    /**
     * Create a new matrix and vector filled with random numbers between the specified values.
     *
     * @param m size of the matrix and vector
     * @param origin lower bound of the values (inclusive)
     * @param bound upper bound of the values (exclusive)
     */
    public MatrixVector(int m, int origin, int bound) {
        // random number generator
        Random rd = new Random();

        // fill matrix and vector with random double values
        double[][] matrix = new double[m][m];
        double[] vector = new double[m];
        for (int i = 0; i < m; i++) {
            vector[i] = (rd.nextInt(bound - origin) + origin)*rd.nextDouble();
            for (int j = 0; j < m; j++) {
                matrix[i][j] = (rd.nextInt(bound - origin) + origin)*rd.nextDouble();
            }
        }

        this.m = m;
        this.matrix = matrix;
        this.vector = vector;
    }

    /**
     * Compute the matrix-vector product sequentially.
     *
     * @return the matrix-vector product
     */
    public double[] sequential() {
        double [] result = new double[this.m];

        // compute matrix-vector product
        for (int i = 0; i < this.m; i++) {
            for (int j = 0; j < this.m; j++) {
                result[i] += this.matrix[i][j]*this.vector[j];
            }
        }

        return result;
    }

    /**
     * Track the execution time of {@link #sequential()}.
     *
     * @return the execution time in milliseconds
     */
    public double timeSequential() {
        long start = System.nanoTime();
        this.sequential();
        long end = System.nanoTime();
        return (end - start)/1e6;
    }




}
