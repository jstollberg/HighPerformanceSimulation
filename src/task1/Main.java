package task1;


import static org.jocl.CL.*;

import java.io.IOException;

import org.jocl.*;

public class Main {
    private static void println(String message)
    {
        System.out.println(message);
    }


    public static void main(String[] args) throws IOException {
        // setup
        int m = 100;
        int origin = -100;
        int bound = 100;

        // compute matrix vector product
        MatrixVector matVec = new MatrixVector(m, origin, bound);
        /* ----------------------------------------------------------------------*/
        double[] sequentialResult = matVec.sequential();
        double seqTime = matVec.timeSequential();

        println("Sequential matrix-vector product took " + seqTime + " ms.");

        /* ----------------------------------------------------------------------*/
        println("Starting OpenCL Initialization...");
        clInit();

        /* ----------------------------------------------------------------------*/
        println("Starting parallel matrix multiplication...");
        double[] parallelResult = matVec.parallel(context, commandQueue);

        /* ----------------------------------------------------------------------*/
        println("Compare results:");
        for(int i = 0; i < m; i++)
        {
            if(Math.abs(parallelResult[i] - sequentialResult[i]) > 1e-5)
            {
                println("Results didn't match! %.2f != %.2f".formatted(parallelResult[i], sequentialResult[i]));
                return;
            }
        }

        println("Results matched. All OK");


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
        // The platform, device type and device number
        // that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the first available platform
        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id[] devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // Create a context for the selected device
        context = clCreateContext(
                contextProperties, numDevices, new cl_device_id[]{device},
                null, null, null);

        String deviceName = getString(devices[0], CL_DEVICE_NAME);
        System.out.printf("CL_DEVICE_NAME: %s\n", deviceName);

        // Create a command-queue for the selected device
        cl_queue_properties properties = new cl_queue_properties();
        commandQueue = clCreateCommandQueueWithProperties(
                context, device, properties, null);
    }

    private static String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }
}
