package task1;

import org.jocl.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.jocl.CL.*;

public class OpenCL {
    private static boolean _initialized;

    public static boolean isInitialized()
    {
        return _initialized;
    }

    private static boolean _hasKernel;
    public static boolean hasKernel()
    {
        return _hasKernel;
    }

    public static cl_context context;
    public static cl_command_queue commandQueue;
    public static cl_program program;
    public static cl_kernel kernel;
    public static cl_device_id device;
    /**
     * Return the device name of a cl_device_id.
     */
    private static String getDeviceName(cl_device_id device)
    {
        // obtain the length of the string that will be queried
        long[] size = new long[1];
        clGetDeviceInfo(device, CL.CL_DEVICE_NAME, 0, null, size);

        // create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int)size[0]];
        clGetDeviceInfo(device, CL.CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);

        // create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

    private static int getDeviceMaxWorkGroupSize(cl_device_id device)
    {
        // obtain the length of the string that will be queried
        long[] size = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, null, size);

        // create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int)size[0]];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, buffer.length, Pointer.to(buffer), null);

        var bbuffer = ByteBuffer.allocate(Long.BYTES);
        bbuffer.put(buffer);
        bbuffer.flip();
        // create a string from the buffer (excluding the trailing \0 byte)
        return bbuffer.getInt();
    }

    private static String[] getSizes(cl_device_id device, int paramName, int numValues) {
        // The size of the returned data has to depend on
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, paramName, (long) Sizeof.size_t * numValues, Pointer.to(buffer), null);
        String[] values = new String[numValues];
        if (Sizeof.size_t == 4) {
            for (int i = 0; i < numValues; i++) {
                values[i] = String.valueOf(buffer.getInt(i * Sizeof.size_t));
            }
        } else {
            for (int i = 0; i < numValues; i++) {
                values[i] = String.valueOf(buffer.getLong(i * Sizeof.size_t));
            }
        }
        return values;
    }

    /**
     * Return a device parameter of type cl_uint or cl_int.
     * @param device The device id.
     * @param paramName Param.
     * @return Integer valued param.
     */
    private static int getDeviceUint(cl_device_id device, int paramName) {
        int[] value = new int[1];
        clGetDeviceInfo(device, paramName, Sizeof.cl_uint, Pointer.to(value), null);
        return value[0];
    }


    /**
     * Initialize jocl procedures.
     */
    public static void init()
    {
        // the platform, device type and device number that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // obtain the first available platform
        int[] numPlatformsArray = new int[5];
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
        device = devices[deviceIndex];

        // create a context for the selected device
        context = clCreateContext(
                contextProperties, numDevices, new cl_device_id[]{device},
                null, null, null);

        // create a command-queue for the selected device
        cl_queue_properties properties = new cl_queue_properties();
        commandQueue = clCreateCommandQueueWithProperties(
                context, device, properties, null);

        _initialized = true;
    }

    /**
     * Releases all data from opencl handles and buffers. After this, OpenCL can not be used!
     */
    public static void release()
    {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);

        _initialized = false;
        _hasKernel = false;
    }

    /**
     * Create kernel for the parallel execution on GPU context.
     *
     * @param kernelname: Name of the kernel
     * @param kernelSource : Source code of kernel
     */
    public static void createKernel(String kernelSource, String kernelname/*, int m*/) {

        // use this in addition to a #define in the kernel
        //kernelSource = kernelSource.replace("{m}", "%d".formatted(m));

        // create the program from the source code
        program = clCreateProgramWithSource(context,
                1, new String[]{ kernelSource }, null, null);

        // build the program
        clBuildProgram(program, 0, null, null, null, null);

        // create the kernel
        kernel = clCreateKernel(program, kernelname, null);

        _hasKernel = true;
    }

    public static void summary()
    {
        Main.header("OPENCL SUMMARY");

        String deviceName = getDeviceName(device);
        System.out.printf("%35s%30s%n", "CL_DEVICE_NAME", deviceName);

        String maxWorkGroupSize = String.valueOf(getDeviceMaxWorkGroupSize(device));
        System.out.printf("%35s%30s%n", "CL_DEVICE_MAX_WORK_GROUP_SIZE", maxWorkGroupSize);

        var maxDim = getDeviceUint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        System.out.printf("%35s%30d%n", "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", maxDim);

        String maxWorkItemSizes = "%s, %s, %s".formatted((Object[]) getSizes(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxDim));
        System.out.printf("%35s%30s%n", "CL_DEVICE_MAX_WORK_ITEM_SIZES", maxWorkItemSizes);

        var maxComputeUnits  = getDeviceUint(device, CL_DEVICE_MAX_COMPUTE_UNITS);
        System.out.printf("%35s%30d%n", "CL_DEVICE_MAX_COMPUTE_UNITS", maxComputeUnits);

        System.out.println("\n".repeat(1));
    }

    public static int getAvailableCUs() {
        return getDeviceUint(device, CL_DEVICE_MAX_COMPUTE_UNITS);
    }
}
