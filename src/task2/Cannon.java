package task2.task2;

import mpi.*;
import java.util.Random;
import java.util.stream.IntStream;

public class Cannon {

    /**
     * Compute a matrix-product based on Cannon's algorithm.
     *
     * @param args
     * @throws MPIException
     */
    public static void main(String[] args) throws MPIException {
        MPI.Init(args);
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        // Initialize matrices A, B, and C
        int nProcesses = Integer.valueOf(args[1]);
        int scaling = Integer.valueOf(args[3]);
        int n = 250 * (int) Math.sqrt(nProcesses) * scaling;
        //int n = 4000;
        short[] A = new short[n * n];
        short[] B = new short[n * n];
        short[] C = new short[n * n];

        Random rd = new Random();
        int origin = 0;
        int bound = 100;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = (short) (rd.nextInt(bound - origin) + origin);
                B[i * n + j] = (short) (rd.nextInt(bound - origin) + origin);
            }
        }

        // Information on matrix and processes
        if (rank == 0) {
            System.out.println("--------------------");
            System.out.println("PROCESSES : " + nProcesses);
            System.out.println("SCALING   : " + scaling);
            System.out.println("DIM       : " + n*n);
            System.out.println("--------------------");
        }

        // Check if matrix has correct size
        if (n % Math.sqrt(size) != 0) {
            if (rank == 0) {
                System.out.println("Matrix size not divisible by number of processes");
            }
            MPI.Finalize();
            return;
        }

        // Set um cartesian communicator
        int[] dims = new int[]{(int) Math.sqrt(size), (int) Math.sqrt(size)};
        boolean[] periods = new boolean[]{true, true};
        Cartcomm comm = MPI.COMM_WORLD.Create_cart(dims, periods, true);
        int[] coords = comm.Coords(rank);

        // Decompose matrices into blocks
        int blockSize = (int) ((double) n / Math.sqrt(size));
        int subSize = blockSize * blockSize;
        short[] localA = new short[subSize];
        short[] localB = new short[subSize];
        short[] localC = new short[subSize];

        // Reorder A and B for the scattering process
        short[] mpiA = new short[n * n];
        short[] mpiB = new short[n * n];
        short[] mpiC = new short[n * n];
        for (int i = 0; i < size; i++) {
            int k = 0;
            for (int j = i * subSize; j < i * subSize + subSize; j++) {
                mpiA[j] = A[comm.Coords(i)[1] * n / dims[1] + comm.Coords(i)[0] * n / dims[0] * n + k];
                mpiB[j] = B[comm.Coords(i)[1] * n / dims[1] + comm.Coords(i)[0] * n / dims[0] * n + k];
                k += 1;
                if (k % blockSize == 0) {
                    k += n - blockSize;
                }
            }
        }

        // Start timing
        comm.Barrier();
        double mpiTime = 0.0;
        if (rank == 0) {
            mpiTime = MPI.Wtime();
        }

        // Scatter blocks of A and B to processes
        /*
        comm.Barrier();
        double scatterTime = 0.0;
        if (rank == 0) {
            scatterTime = MPI.Wtime();
        }
         */

        MPI.COMM_WORLD.Scatter(mpiA, 0, subSize, MPI.SHORT, localA, 0, subSize, MPI.SHORT, 0);
        MPI.COMM_WORLD.Scatter(mpiB, 0, subSize, MPI.SHORT, localB, 0, subSize, MPI.SHORT, 0);

        /*
        comm.Barrier();
        if (rank == 0) {
            scatterTime = MPI.Wtime() - scatterTime;
            System.out.println("Took " + scatterTime + " s for scattering");
        }
         */

        // Shift on A
        ShiftParms parms = comm.Shift(1, -coords[0]);
        int src = parms.rank_source;
        int dest = parms.rank_dest;
        MPI.COMM_WORLD.Sendrecv_replace(localA, 0, subSize, MPI.SHORT, dest, 0, src, 0);

        // Shift on B
        parms = comm.Shift(0, -coords[1]);
        src = parms.rank_source;
        dest = parms.rank_dest;
        MPI.COMM_WORLD.Sendrecv_replace(localB, 0, subSize, MPI.SHORT, dest, 0, src, 0);

        // C_ij = A_ij*B_ji
        Request[] req = new Request[4];
        for (int l = 0; l < (int) Math.sqrt(size); l++) {

            // Perform local matrix multiplication
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    for (int k = 0; k < blockSize; k++) {
                        localC[i * blockSize + k] += localA[i * blockSize + j] * localB[j * blockSize + k];
                    }
                }
            }

            // Shift on A
            parms = comm.Shift(1, -1);
            src = parms.rank_source;
            dest = parms.rank_dest;
            //MPI.COMM_WORLD.Sendrecv_replace(localA, 0, subSize, MPI.SHORT, dest, 0, src, 0);  // this is blocking
            req[0] = MPI.COMM_WORLD.Isend(localA, 0, subSize, MPI.SHORT, dest, 0);
            req[1] = MPI.COMM_WORLD.Irecv(localA, 0, subSize, MPI.SHORT, src, 0);

            // Shift on B
            parms = comm.Shift(0, -1);
            src = parms.rank_source;
            dest = parms.rank_dest;
            //MPI.COMM_WORLD.Sendrecv_replace(localB, 0, subSize, MPI.SHORT, dest, 0, src, 0);  // this is blocking
            req[2] = MPI.COMM_WORLD.Isend(localB, 0, subSize, MPI.SHORT, dest, 0);
            req[3] = MPI.COMM_WORLD.Irecv(localB, 0, subSize, MPI.SHORT, src, 0);

            // Wait until all requests are finished
            for (Request r : req) {
                r.Wait();
            }

        }

        // Gather blocks of C from processes
        /*
        comm.Barrier();
        double gatherTime = 0.0;
        if (rank == 0) {
            gatherTime = MPI.Wtime();
        }
         */
        MPI.COMM_WORLD.Gather(localC, 0, subSize, MPI.SHORT, mpiC, 0, subSize, MPI.SHORT, 0);
        comm.Barrier();
        /*
        if (rank == 0) {
            gatherTime = MPI.Wtime() - gatherTime;
            System.out.println("Took " + gatherTime + " s for gathering");
        }
         */

        // Stop timing
        comm.Barrier();
        if (rank == 0) {
            mpiTime = MPI.Wtime() - mpiTime;
            System.out.println("MPI    : " + mpiTime + " s");
        }

        // Restore original order of C
        for (int i = 0; i < size; i++) {
            int k = 0;
            for (int j = i * subSize; j < i * subSize + subSize; j++) {
                C[comm.Coords(i)[1] * n / dims[1] + comm.Coords(i)[0] * n / dims[0] * n + k] = mpiC[j];
                k += 1;
                if (k % blockSize == 0) {
                    k += n - blockSize;
                }
            }
        }

        if (rank == 0) {
            // Compute result serially for comparison
            short[] serialC = new short[n*n];
            double serialTime = MPI.Wtime();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        serialC[i * n + k] += A[i * n + j] * B[j * n + k];
                    }
                }
            }
            serialTime = MPI.Wtime() - serialTime;
            System.out.println("SERIAL : " + serialTime + " s");

            // Compute result via parallel stream
            short[] streamC = new short[n * n];
            double streamTime = MPI.Wtime();
            IntStream.range(0, n).parallel().forEach(i -> {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        streamC[i * n + k] += A[i * n + j] * B[j * n + k];
                    }
                }
            });
            streamTime = MPI.Wtime() - streamTime;
            System.out.println("STREAM : " + streamTime + " s");

            // Check results
            System.out.println("--------------------");
            if (checkEquality(serialC, C) && checkEquality(streamC, C)) {
                System.out.println("OK");
            } else {
                System.out.println("RESULTS DO NOT MATCH");
            }
        }

        MPI.Finalize();

    }

    /**
     * Check if two 1d arrays are equal.
     *
     * @param A the first array
     * @param B the second array
     * @return true if arrays are equal, else false
     */
    private static boolean checkEquality(short[] A, short[] B) {
        for (int i = 0; i < A.length; i++) {
            if (A[i] - B[i] > 1e-3 || A[i] - B[i] < -1e-3) {
                return false;
            }
        }
        return true;
    }

    /**
     * Print a matrix stored in a 1d array.
     *
     * @param mat the matrix to print
     */
    private static void printMatrix(short[] mat)
    {
        var n = Math.sqrt(mat.length);
        StringBuilder output = new StringBuilder();
        for(int i = 0; i < n*n; i++)
        {
            output.append("%6d ".formatted(mat[i]));

            if ((i + 1) % n == 0)
                output.append("\n");
        }
        System.out.println(output);
    }

}