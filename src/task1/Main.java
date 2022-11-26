package task1;

import java.util.Arrays;

class TestResult{
    /**
     * Results from sequential multiplication.
     */
    double[] sresults;
    /**
     * results from parallel multiplication.
     */
    double[] presults;
    /**
     * Used matrix size.
     */
    int m;
    /**
     * Value for local work size.
     */
    long local_work_size;

    /**
     * Calculate average sequential time.
     * @return Average sequential time.
     */
    public double getAvgSequentialTime()
    {
        return Arrays.stream(sresults).parallel().average().orElse(0);
    }

    /**
     * Calculate average parallel time.
     * @return Average parallel time.
     */
    public double getAvgParallelTime()
    {
        return Arrays.stream(presults).parallel().average().orElse(0);
    }

    TestResult(double[][] results, int m, long lws)
    {
        sresults = results[0];
        presults = results[1];
    }
}


public class Main {

    private static void println(String message)
    {
        System.out.println(message);
    }
    private static void print(String message)
    {
        System.out.print(message);
    }
    public static void main(String[] args)  {
        // setup
        int m;
        long local_work_size;

        if(args.length > 1) {
            m = Integer.parseInt(args[0]);
            local_work_size = Long.parseLong(args[1]);

            runTests(m, local_work_size);

            OpenCL.release();
            return;
        }
        println("RUNNING TEST SUITE...");

        int[] matrix_sizes = new int[]{500,5000,10000, 20000, 40000};
        long[] local_work_sizes = new long[]{1,2,5,10,20,-1};

        /* ----------------------------------------------------------------------*/
        println("Starting OpenCL Initialization...");
        OpenCL.init();

        /* ----------------------------------------------------------------------*/
        println("Starting test runs...");
        var times = new TestResult[matrix_sizes.length][local_work_sizes.length];

        for (int i = 0; i < matrix_sizes.length; i++) {
            int matrix_size = matrix_sizes[i];
            for (int j = 0; j < local_work_sizes.length; j++) {
                long localWorkSize = local_work_sizes[j];

                try {
                    var output = runTests(matrix_size, localWorkSize);

                    times[i][j] = new TestResult(output, matrix_size, localWorkSize);
                } catch (Exception e) {
                    println("ERROR -- m: %d, local_work_size: %d | %s".formatted(matrix_size, localWorkSize, e.toString()));
                }
            }
        }
        printResults(times, matrix_sizes, local_work_sizes);

        OpenCL.release();
    }

    private static void printResults(TestResult[][] results, int[] sizes, long[] workSizes)
    {
        var savgs = new double[sizes.length][workSizes.length];
        var pavgs = new double[sizes.length][workSizes.length];

        var pad = "                        ";

        var format = "Matrix \\ Local-WorkSize" + "%11s             ".repeat(workSizes.length);
        println(String.format(format, (Object[]) toArray(workSizes)));

        for (int i = 0; i < sizes.length; i++) {
            int matrix_size = sizes[i];
            var line = "        %9d       ".formatted(matrix_size);
            var avgs = new String[workSizes.length];
            for (int j = 0; j < workSizes.length; j++) {
                long localWorkSize = workSizes[j];
                var result = results[i][j];

                savgs[i][j] = result.getAvgSequentialTime();
                pavgs[i][j] = result.getAvgParallelTime();

                avgs[j] = "%8.2f / %8.2f".formatted(savgs[i][j],pavgs[i][j]);
            }
            line = line + "%16s     ".repeat(workSizes.length).formatted((Object[]) avgs);
            println(line);
        }
    }

    private static String red(String input)
    {
        return "\u001B[31m" + input + "\u001B[0m";
    }
    private static String green(String input)
    {
        return "\u001B[32m" + input + "\u001B[0m";
    }
    private static String[] toArray(long[] longs)
    {
        String[] strArray = new String[longs.length];

        for (int i = 0; i < longs.length; i++) {
            strArray[i] = String.valueOf(longs[i]);

        }

        return strArray;
    }


    private static double[][] runTests(int m, long local_work_size) {
        int origin = -10;
        int bound = 11;

        // number of sequential executions
        int sIter = 10;
        // number of parallel executions
        int pIter = 10;

        // return timings
        var returns = new double[2][];
        returns[0] = new double[sIter];
        returns[1] = new double[pIter];

        /* ----------------------------------------------------------------------*/
        println("------------------------------------------");
        println("Running matrix calculations with:");
        println("\tm=%d".formatted(m));
        println("\tlocal_work_size=%d".formatted(local_work_size));
        println("\tWork-groups=%d".formatted(m / local_work_size));
        println("");
        println("");

        final int mm = m;
        println("Create matrix-vector structure...");
        var creation = Timings.time( () -> new MatrixVector(mm, origin, bound));
        println("\t> Took " + creation.time + " ms.");

        MatrixVector matVec = creation.result;

        /* ----------------------------------------------------------------------*/
        println("Initializing parallel matrix multiplication...");
        matVec.initParallel(local_work_size);

        /* ----------------------------------------------------------------------*/
        println("Running sequential multiplication...");
        TimeResult<short[]> sequentialResult = null;
        // execute real timings
        for(int i = 0; i < sIter; i++)
        {
            print("\t[%2d|%2d]: ".formatted(i+1, sIter));
            sequentialResult = Timings.time(matVec::sequential);
            println("%8.2fms".formatted(sequentialResult.time));

            returns[0][i] = sequentialResult.time;
        }

        /* ----------------------------------------------------------------------*/
        println("Running parallel multiplication...");
        // execute real timings
        for(int i = 0; i < pIter; i++)
        {
            print("\t[%2d|%2d]: ".formatted(i+1, pIter));
            var parallelTime = Timings.time(matVec::parallel);
            println("%8.2fms".formatted(parallelTime));

            var parallelResult = matVec.readParallel();
            if (areResultsEqual(m, sequentialResult.result, parallelResult)){
                returns[1][i] = parallelTime;
            }else{
                throw new RuntimeException("Parallel result did not match!");
            }
        }
        println("Cleaning parallel...");
        matVec.releaseParallel();

        return returns;
    }

    private static boolean areResultsEqual(int m, short[] sequential, short[] parallel) {
        for(int i = 0; i < m; i++)
        {
            if(Math.abs(sequential[i] - parallel[i]) > 1e-5)
            {
                println("\t> Results didn't match! %d != %d".formatted(sequential[i], parallel[i]));
                return false;
            }
        }
        return true;
    }
}

