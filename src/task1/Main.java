package task1;

import static org.jocl.CL.*;

import java.io.IOException;
import org.jocl.*;

interface ResultFunction
{
    short[] run();
}

interface ExecuteFunction
{
    void run();
}

interface ResultAnyFunction<T>
{
    T run();
}

class TimeResult<T>
{
    double time;
    T result;

    TimeResult(double time, T result)
    {
        this.time = time;
        this.result = result;
    }
}

public class Main {

    private static void println(String message)
    {
        System.out.println(message);
    }

    /**
     * Use this with a lambda function to time a function call.
     *
     * @param func: The function to be timed.
     * @return A result, which contains a short array and the time it took to process.
     */
    private static TimeResult time(ResultFunction func)
    {
        long start = System.nanoTime();
        short[] result = func.run();
        long end = System.nanoTime();
        double time =  (end - start)/1e6;

        return new TimeResult(time, result);
    }

    /**
     * Time a function call.
     *
     * @param resultFunc The function to get the results from.
     * @param timedFunc The function to time.
     * @return A result containing time of timedFunc and results of resultFunc.
     */
    public static TimeResult time(ResultFunction resultFunc, ExecuteFunction timedFunc)
    {
        long start = System.nanoTime();
        timedFunc.run();
        long end = System.nanoTime();
        double time =  (end - start)/1e6;

        return new TimeResult(time, resultFunc.run());
    }

    /**
     * Time a function call.
     *
     * @param resultFunc The function to get the results from.
     * @return A result containing time of timedFunc and results of resultFunc.
     */
    public static TimeResult time(ResultAnyFunction resultFunc)
    {
        long start = System.nanoTime();
        resultFunc.run();
        long end = System.nanoTime();
        double time =  (end - start)/1e6;

        return new TimeResult(time, resultFunc.run());
    }

    public static void main(String[] args) throws IOException {
        // setup
        int m = 10000;
        long local_work_size = 1;
        long heapsize = Runtime.getRuntime().totalMemory();
        System.out.println("heapsize is :: " + heapsize);
        if(args.length > 1) {
            m = Integer.parseInt(args[0]);
            local_work_size = Long.parseLong(args[1]);
        }else{
            println("Supply arguments: [matrix size: int] [local_worker_size: long (-1 for null)]");
            return;
        }


        int origin = -10;
        int bound = 11;

        /* ----------------------------------------------------------------------*/
        println("Running matrix calculations with:");
        println("\tm=%d".formatted(m));
        println("\tlocal_work_size=%d".formatted(local_work_size));
        println("\tWork-groups=%d".formatted(m/local_work_size));
        println("");
        println("");

        final int mm = m;
        println("Create matrix-vector structure...");
        TimeResult<MatrixVector> creation = time( () -> new MatrixVector(mm, origin, bound));
        println("\t> Took " + creation.time + " ms.");

        MatrixVector matVec = creation.result;


        /* ----------------------------------------------------------------------*/
        // sequential calculation
        println("Running sequential matrix multiplication...");
        TimeResult<short[]> sequentialResult = time(matVec::sequential);
        println("\t> Took " + sequentialResult.time + " ms.");

        /* ----------------------------------------------------------------------*/
        println("Starting OpenCL Initialization...");
        clInit();

        /* ----------------------------------------------------------------------*/
        println("Initializing parallel matrix multiplication...");
        matVec.initParallel(context, local_work_size);
        // warmup to load kernel onto gpu
        println("Running warmup...");
        matVec.parallel(commandQueue);
        matVec.parallel(commandQueue);
        // time imax times
        int imax = 5;
        for(int i = 0; i < imax; i++)
        {
            println("Running parallel (%d of %d)...".formatted(i+1, imax));
            TimeResult<short[]> parallelResult = time(
                    () -> matVec.readParallel(commandQueue),
                    () -> matVec.parallel(commandQueue));
            println("\t> Took " + parallelResult.time + " ms.");

            if (!areResultsEqual(m, sequentialResult.result, parallelResult.result)) return;
        }

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
        println("\t> Results match!");
        return true;
    }


    private static cl_context context;
    private static cl_command_queue commandQueue;

    /**
     * Initialize kernel and context.
     */
    public static void clInit()
    {
        // initialize gpu with kernel source
        defaultInitialization();
    }

    private static void defaultInitialization()
    {
        // the platform, device type and device number that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // obtain the first available platform
        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // obtain a platform ID
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // obtain a device ID
        cl_device_id[] devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // create a context for the selected device
        context = clCreateContext(
                contextProperties, numDevices, new cl_device_id[]{device},
                null, null, null);

        String deviceName = getString(devices[0], CL_DEVICE_NAME);
        System.out.printf("CL_DEVICE_NAME: %s\n", deviceName);

        // create a command-queue for the selected device
        cl_queue_properties properties = new cl_queue_properties();
        commandQueue = clCreateCommandQueueWithProperties(
                context, device, properties, null);
    }

    private static String getString(cl_device_id device, int paramName)
    {
        // obtain the length of the string that will be queried
        long[] size = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }
}
