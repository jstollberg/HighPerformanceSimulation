package task1;

import java.util.Arrays;
import java.util.LinkedList;

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

    TimedResults(double[][] results)
    {
        sresults = results[0]; // sequential results
        presults = results[1]; // parallel results
    }
}

/**
 * Wrapper class around test run results.
 */
class TestResult
{
    /**
     * Time measurements of test run.
     */
    public TimedResults times;
    /**
     * The matrixvector structure used. This is only used as a cache object and will be overriden during testing.
     * Every TestResult of a series of different local_work_sizes will contain a reference to the same matrixvector!
     */
    public MatrixVector cachedMatrixVector;

    private final int m; // matrix size
    private final long local_work_size; // local_work_size

    public int getMatrixSize()
    {
        return this.m;
    }

    public long getLocalWorkSize()
    {
        return local_work_size;
    }

    /**
     * On successful use this to create test results.
     * @param _times Measurements. [0][..] sequential, [1][..] parallel
     * @param vector The measured matrixvector.
     */
    public TestResult(double[][] _times, MatrixVector vector)
    {
        times = new TimedResults(_times);
        cachedMatrixVector = vector;

        // we need to extract these values, because matVec will still be in use afterwards!
        m = cachedMatrixVector.getMatrixSize();
        local_work_size = cachedMatrixVector.getLocalWorkSize();
    }

    /**
     * If the test failed, this contains the exception.
     */
    public Exception error;

    /**
     * Use this to create a failed test result!
     * @param matrix_size Matrix size
     * @param localWorkSize lws
     * @param e The exception which occured during the test.
     */
    public TestResult(int matrix_size, long localWorkSize, Exception e) {
        times = new TimedResults(new double[2][0]);
        error = e;

        m = matrix_size;
        local_work_size = localWorkSize;
    }
}



public class Main {
    private static boolean disableOutput = false;
    private static void println(String message)
    {
        if(!disableOutput)
            System.out.println  (message);
    }
    private static void print(String message)
    {
        if(!disableOutput)
            System.out.print(message);
    }
    public static void main(String[] args)  {
        // setup
        int m;
        long local_work_size;

        // disable detailed output of test runs
        boolean disableTestRunOutput = true;

        // parse command line arguments, if any
        if(args.length > 1) {
            m = Integer.parseInt(args[0]);
            local_work_size = Long.parseLong(args[1]);

            runTests(m, local_work_size, null);

            OpenCL.release();
            return;
        }

        /* ----------------------------------------------------------------------*/
        /* ----------------------    STANDALONE RUNNER   ------------------------*/
        /* ----------------------------------------------------------------------*/


        println("RUNNING TEST SUITE...");

        int[] matrix_sizes = new int[]{10,1000,2000,4000,8000,15000,40000};
        long[] local_work_sizes = new long[]{1,2,5,10,20,50,1000,-1};

        /* ----------------------------------------------------------------------*/
        println("Starting OpenCL Initialization...");
        OpenCL.init();

        /* ----------------------------------------------------------------------*/
        println("Starting test runs...");
        // matrix of all TimedResults of dimension [matrix_size*local_work_sizes]
        var results = new TestResult[matrix_sizes.length][local_work_sizes.length];
        // these testresults are used in runTests: They will be reset for every matrix size but may be
        //  reused when the matrix size has not changed
        TestResult lastResult = null;

        for (int i = 0; i < matrix_sizes.length; i++) {
            int matrix_size = matrix_sizes[i];
            if(disableTestRunOutput){
                println("Running tests for (m: %8d)...".formatted(matrix_size));
                disableOutput = true;
            }
            for (int j = 0; j < local_work_sizes.length; j++) {
                long localWorkSize = local_work_sizes[j];

                // wrap the test run in a try catch so the whole suite does not break
                try
                {
                    if(disableTestRunOutput){
                        disableOutput = false;
                        print("\t > lws: %5d...".formatted(localWorkSize));
                        disableOutput = true;
                    }


                    // save test results for next test run
                    lastResult = runTests(matrix_size, localWorkSize, lastResult);
                    // create timed results from testResults so we can display them after suite finished
                    results[i][j] = lastResult;

                    disableOutput = false;
                    println(green("SUCCESS"));
                }
                catch(Exception e)
                {
                    if(!disableTestRunOutput)
                        println(red("ERROR") + "m: %d, lws: %d | %s".formatted(matrix_size, localWorkSize, e.toString()));
                    else{
                        disableOutput = false;
                        println(red("FAIL"));
                    }
                    results[i][j] = new TestResult(matrix_size, localWorkSize, e);
                }


            }
            // reset last test run because matrix size will change
            lastResult = null;
        }

        // print timed results
        printResults(results, matrix_sizes, local_work_sizes);


        OpenCL.summary();
        // release all opencl buffers and objects
        OpenCL.release();
    }

    /* ###############################################################*/
    /* #################   RESULT PRINTING           #################*/
    /* ###############################################################*/

    /**
     * Prints results in a table to the console.
     * @param results Timed results. [0][.] for sequential times, [1][.] for parallel times.
     * @param sizes The used matrix sizes.
     * @param workSizes The used local_work_sizes.
     */
    private static void printResults(TestResult[][] results, int[] sizes, long[] workSizes)
    {
        header("TIMING RESULTS");

        // list errors in here during print
        var errors = new LinkedList<String>();

        var format = "%21s%21s".formatted("Matrix \\ Local-WorkSize", "Sequential") + "%21s".repeat(workSizes.length);
        println(String.format(format, (Object[]) toArray(workSizes)));

        for (int i = 0; i < sizes.length; i++) {
            int matrix_size = sizes[i];

            // check sequential results first
            String averageSequentialTime;
            if(results[i][0].error == null)
                averageSequentialTime = "%.2f".formatted(results[i][0].times.getAvgSequentialTime());
            else {
                averageSequentialTime = red("ERR " + errors.size());
                errors.add(String.format("ERR %d (m: %d, lws: -): %s", errors.size(), i, results[i][0].error.getMessage()));
            }
            var line = "%21d          %13s".formatted(matrix_size, averageSequentialTime);

            // avgs contains a string array of parallel average timings for each local_work_size
            var avgs = new String[workSizes.length];
            for (int j = 0; j < workSizes.length; j++) {
                var result = results[i][j];

                if(result.error != null) {
                    avgs[j] = red("ERR " + errors.size());
                    errors.add(String.format("ERR %d (m: %d, lws: %d): %s", errors.size(), result.getMatrixSize(), result.getLocalWorkSize(), result.error.getMessage()));
                }else{
                    avgs[j] = ("%11.2f"+gray(" (%d)")).formatted(result.times.getAvgParallelTime(), result.getLocalWorkSize());
                }
            }
            line = line + "%30s".repeat(workSizes.length).formatted((Object[]) avgs);
            println(line);
        }

        println("Errors:");
        for(String err : errors)
            println(err);
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
    private static TestResult runTests(final int m, long local_work_size, TestResult lastTestResults) {
        int origin = -10;
        int bound = 11;

        // number of sequential executions
        int sIter = 10;
        // number of parallel executions
        int pIter = 10;

        // return timings
        var measuredTimes = new double[2][];
        measuredTimes[0] = new double[sIter]; // sequential timings
        measuredTimes[1] = new double[pIter]; // parallel timings

        /* ----------------------------------------------------------------------*/
        println("------------------------------------------");
        println("Running matrix calculations with:");
        println("\tm=%d".formatted(m));
        println("\tlocal_work_size=%d".formatted(local_work_size));
        println("\tWork-groups=%d".formatted(m / local_work_size));
        println("");
        println("");

        MatrixVector matVec;
        if(lastTestResults == null){
            println(yellow("Create") + " matrix-vector structure...");
            var creation = Timings.time( () -> new MatrixVector(m, origin, bound));
            println("\t> Took " + creation.time + " ms.");

            matVec = creation.result;
        }
        else
        {
            println(green("Reusing") +" last MatrixVector...");
            matVec = lastTestResults.cachedMatrixVector;
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

                measuredTimes[0][i] = sequentialResult.time;
            }
        }else
        {
            println(green("Reusing") +" last sequential results...");
            measuredTimes[0] = lastTestResults.times.sresults;
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
                measuredTimes[1][i] = parallelTime;
            }else{
                throw new RuntimeException("Parallel result did not match!");
            }
        }
        println("Release local parallel buffers.");
        matVec.releaseParallel();

        return new TestResult(measuredTimes, matVec);
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

    public static void header(String header) {
        println("");
        println("");
        println(" ############%s############".formatted(centerString(30,header)));
        println("");
    }

    private static String centerString (int width, String s) {
        return String.format("%-" + width  + "s", String.format("%" + (s.length() + (width - s.length()) / 2) + "s", s));
    }
}

