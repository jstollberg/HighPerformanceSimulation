package task2;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * Class wrapping a matrix lhs and a vector rhs.
 */
public class MatrixMatrix {
    int m;
    short[] A;
    short[] B;
    short[] C;
    short[] solution;
    /**
     * Create new matrices filled with random numbers between the specified values.
     *
     * @param m size of the matrices
     * @param origin lower bound of the values (inclusive)
     * @param bound upper bound of the values (exclusive)
     */
    public MatrixMatrix(int m, int origin, int bound) {
        // random number generator
        Random rd = new Random();

        // fill matrices with random values
        A = new short[m * m];
        B = new short[m * m];
        C = new short[m * m];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                A[i * m + j] = (short) (rd.nextInt(bound - origin) + origin);
                B[i * m + j] = (short) (rd.nextInt(bound - origin) + origin);
                C[i * m + j] = (short) (rd.nextInt(bound - origin) + origin);
            }
        }

        this.m = m;
    }

    /**
     * Compute the matrix product C=AB+C sequentially.
     *
     * @return the matrix product
     */
    public short[] sequential() {
        solution = new short[m * m];

        // compute matrix-vector product
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < m; k++) {
                    solution[i * m + k] += A[i * m + j] * B[j * m + k] + C[i * m + k];
                }
            }
        }

        return solution;
    }

    /* ###############################################################*/
    /* #################   PARALLEL PROCEDURES       #################*/
    /* ###############################################################*/

    short[] streamSolution;

    /**
     * Perform the computation of the matrix product using a parallel stream.
     *
     * @return the matrix product
     */
    public short[] parallelAsStream() {
        streamSolution = new short[m * m];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < m; k++) {
                    streamSolution[i * m + k] += A[i * m + j] * B[j * m + k] + C[i * m + k];
                }
            }
        });
        return streamSolution;
    }

}
