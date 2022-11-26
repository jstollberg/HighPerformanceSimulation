package task1;

interface ShortResultFunction
{
    short[] run();
}

interface ExecuteFunction
{
    void run();
}

interface AnyResultFunction<T>
{
    T run();
}


public class Timings {

    /**
     * Use this with a lambda function to time a function call.
     *
     * @param func: The function to be timed.
     * @return A result, which contains a short array and the time it took to process.
     */
    public static TimeResult<short[]> time(ShortResultFunction func)
    {
        long start = System.nanoTime();
        short[] result = func.run();
        long end = System.nanoTime();
        double time =  (end - start)/1e6;

        return new TimeResult<>(time, result);
    }

    /**
     * Time a function without return type.
     * @param func
     * @return
     */
    public static double time(ExecuteFunction func)
    {
        long start = System.nanoTime();
        func.run();
        long end = System.nanoTime();
        return (end - start)/1e6;
    }



    /**
     * Time a function call.
     *
     * @param resultFunc The function to get the results from.
     * @param timedFunc The function to time.
     * @return A result containing time of timedFunc and results of resultFunc.
     */
    public static TimeResult<short[]> time(ShortResultFunction resultFunc, ExecuteFunction timedFunc)
    {
        long start = System.nanoTime();
        timedFunc.run();
        long end = System.nanoTime();
        double time =  (end - start)/1e6;

        return new TimeResult<>(time, resultFunc.run());
    }

    /**
     * Time a function call.
     *
     * @param resultFunc The function to get the results from.
     * @return A result containing time of timedFunc and results of resultFunc.
     */
    public static <T> TimeResult<T> time(AnyResultFunction<T> resultFunc)
    {
        long start = System.nanoTime();
        resultFunc.run();
        long end = System.nanoTime();
        double time =  (end - start)/1e6;

        return new TimeResult<>(time, resultFunc.run());
    }


}
