package task1;

import org.jocl.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

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
        cl_device_id device = devices[deviceIndex];

        // create a context for the selected device
        context = clCreateContext(
                contextProperties, numDevices, new cl_device_id[]{device},
                null, null, null);

        String deviceName = getDeviceName(devices[0]);
        System.out.printf("CL_DEVICE_NAME: %s\n", deviceName);
        String maxWorkGroupSize = String.valueOf(getDeviceMaxWorkGroupSize(devices[0]));
        System.out.printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %s\n", maxWorkGroupSize);

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
     * @param name: The name of the kernel (file and kernel name).
     * @throws IOException : matrix_vec.cl not found.
     */
    public static void createKernel(String name/*, int m*/) throws IOException {
        String kernelSource = Files.readString(Path.of("%s.cl".formatted(name)));

        // use this in addition to a #define in the kernel
        //kernelSource = kernelSource.replace("{m}", "%d".formatted(m));

        // create the program from the source code
        program = clCreateProgramWithSource(context,
                1, new String[]{ kernelSource }, null, null);

        // build the program
        clBuildProgram(program, 0, null, null, null, null);

        // create the kernel
        kernel = clCreateKernel(program, name, null);

        _hasKernel = true;
    }
}
