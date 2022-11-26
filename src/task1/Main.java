package task1;

import java.util.Arrays;

/**
 * Result Wrapper containing the timing results for both sequential and parallel executions.
 */
class TimedResults {
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
        return Arrays.stream(presults).skip(1).parallel().average().orElse(0);
    }

    TimedResults(double[][] results, int m, long lws)
    {
        sresults = results[0]; // sequential results
        presults = results[1]; // parallel results
        this.m = m;
        local_work_size = lws;
    }
}

/**
 * Wrapper class around test run results.
 */
class TestResults
{
    public TimedResults times;
    public MatrixVector matVec;
    public TestResults(double[][] _times, MatrixVector vector, int m, long local_work_size)
    {
        times = new TimedResults(_times, m, local_work_size);
        matVec = vector;
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

            runTests(m, local_work_size, null);

            OpenCL.release();
            return;
        }
        println("RUNNING TEST SUITE...");

        int[] matrix_sizes = new int[]{10,1000,2000,4000,8000,15000};
        long[] local_work_sizes = new long[]{1,2,5,10,20,50,-1};

        /* ----------------------------------------------------------------------*/
        println("Starting OpenCL Initialization...");
        OpenCL.init();

        /* ----------------------------------------------------------------------*/
        println("Starting test runs...");
        // matrix of all TimedResults of dimension [matrix_size*local_work_sizes]
        var times = new TimedResults[matrix_sizes.length][local_work_sizes.length];
        // these testresults are used in runTests: They will be reset for every matrix size but may be
        //  reused when the matrix size has not changed
        TestResults testResults = null;

        for (int i = 0; i < matrix_sizes.length; i++) {
            int matrix_size = matrix_sizes[i];
            for (int j = 0; j < local_work_sizes.length; j++) {
                long localWorkSize = local_work_sizes[j];

                // wrap the test run in a try catch so the whole suite does not break
                try
                {
                    // save test results for next test run
                    testResults = runTests(matrix_size, localWorkSize, testResults);
                    // create timed results from testResults so we can display them after suite finished
                    times[i][j] = testResults.times;
                }
                catch(Exception e)
                {
                    println(red("ERROR") + "m: %d, lws: %d | %s".formatted(matrix_size, localWorkSize, e.toString()));
                }

            }
            // reset last test run because matrix size will change
            testResults = null;
        }

        // print timed results
        printResults(times, matrix_sizes, local_work_sizes);


        OpenCL.summary();
        // release all opencl buffers and objects
        OpenCL.release();
    }

    /**
     * Prints results in a table to the console.
     * @param results Timed results. [0][.] for sequential times, [1][.] for parallel times.
     * @param sizes The used matrix sizes.
     * @param workSizes The used local_work_sizes.
     */
    private static void printResults(TimedResults[][] results, int[] sizes, long[] workSizes)
    {
        var pavgs = new double[sizes.length][workSizes.length];

        var pad = "                        ";

        var format = "%21s%21s".formatted("Matrix \\ Local-WorkSize", "Sequential") + "%21s".repeat(workSizes.length);
        println(String.format(format, (Object[]) toArray(workSizes)));

        for (int i = 0; i < sizes.length; i++) {
            int matrix_size = sizes[i];
            var line = "%21d               %8.2f".formatted(matrix_size, results[i][0].getAvgSequentialTime());
            var avgs = new String[workSizes.length];
            for (int j = 0; j < workSizes.length; j++) {
                long localWorkSize = workSizes[j];
                var result = results[i][j];


                pavgs[i][j] = result.getAvgParallelTime();

                avgs[j] = ("%11.2f"+gray(" (%d)")).formatted(pavgs[i][j], result.local_work_size);
            }
            line = line + "%30s".repeat(workSizes.length).formatted((Object[]) avgs);
            println(line);
        }
    }

    /* ###############################################################*/
    /* #################   CONSOLE MODS              #################*/
    /* ###############################################################*/
    private static String red(String input)
    {
        return "\u001B[31m" + input + "\u001B[0m";
    }
    private static String green(String input)
    {
        return "\u001B[32m" + input + "\u001B[0m";
    }
    private static String yellow(String input)
    {
        return "\u001B[33m" + input + "\u001B[0m";
    }
    private static String gray(String input)
    {
        return "\u001B[90m" + input + "\u001B[0m";
    }
    private static String[] toArray(long[] longs)
    {
        String[] strArray = new String[longs.length];

        for (int i = 0; i < longs.length; i++) {
            strArray[i] = String.valueOf(longs[i]);

        }
        return strArray;
    }


    /* ###############################################################*/
    /* #################   TESTING FUNCTIONS         #################*/
    /* ###############################################################*/

    /**
     * Run a test using supplied parameters.
     * @param m The size of the matrix.
     * @param local_work_size OpenCL argument.
     * @param lastTestResults Last test result. Used to skip unnecessary creation and sequential multiplication calls.
     * @return A TestResult containing times and the generated MatrixVector class. If null, the MatrixVector is created
     *          and the sequential calculation is performed.
     */
    private static TestResults runTests(final int m, long local_work_size, TestResults lastTestResults) {
        int origin = -10;
        int bound = 11;

        // number of sequential executions
        int sIter = 10;
        // number of parallel executions
        int pIter = 10;

        // return timings
        var returns = new double[2][];
        returns[0] = new double[sIter]; // sequential timings
        returns[1] = new double[pIter]; // parallel timings

        /* ----------------------------------------------------------------------*/
        println("------------------------------------------");
        println("Running matrix calculations with:");
        println("\tm=%d".formatted(m));
        println("\tlocal_work_size=%d".formatted(local_work_size));
        println("\tWork-groups=%d".formatted(m / local_work_size));
        println("");
        println("");

        MatrixVector matVec = null;
        if(lastTestResults == null){
            println(yellow("Create") + " matrix-vector structure...");
            var creation = Timings.time( () -> new MatrixVector(m, origin, bound));
            println("\t> Took " + creation.time + " ms.");

            matVec = creation.result;
        }
        else
        {
            println(green("Reusing") +" last MatrixVector...");
            matVec = lastTestResults.matVec;
        }

        /* ----------------------------------------------------------------------*/
        println("Initializing parallel matrix multiplication...");
        matVec.initParallel(local_work_size);

        /* ----------------------------------------------------------------------*/
        TimeResult<short[]> sequentialResult = null;
        if(lastTestResults == null)
        {
            println(yellow("Running") +" sequential multiplication...");
            // execute real timings
            for(int i = 0; i < sIter; i++)
            {
                print("\t[%2d|%2d]: ".formatted(i + 1, sIter));
                sequentialResult = Timings.time(matVec::sequential);
                println("%8.2fms".formatted(sequentialResult.time));

                returns[0][i] = sequentialResult.time;
            }
        }else
        {
            println(green("Reusing") +" last sequential results...");
            returns[0] = lastTestResults.times.sresults;
            sequentialResult = new TimeResult<>(0, matVec.solution);
        }

        /* ----------------------------------------------------------------------*/
        println(yellow("Running") +" parallel multiplication...");
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
        println("Release local parallel buffers.");
        matVec.releaseParallel();

        return new TestResults(returns, matVec, m, matVec.getLocalWorkSize());
    }

    /**
     * Compare two results.
     * @param m Size of results.
     * @param sequential .
     * @param parallel .
     * @return .
     */
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

