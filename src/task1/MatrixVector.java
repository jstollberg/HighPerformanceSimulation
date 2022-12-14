package task1;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.jocl.CL.*;

/**
 * Class wrapping a matrix lhs and a vector rhs.
 */
public class MatrixVector {
    // create memory objects
    private cl_mem[] memBuffers;

    int m;
    // left hand side
    short[] matrix;
    // right hand side
    short[] vector;

    short[] solution;
    /**
     * Create a new matrix and vector filled with random numbers between the specified values.
     *
     * @param m size of the matrix and vector
     * @param origin lower bound of the values (inclusive)
     * @param bound upper bound of the values (exclusive)
     */
    public MatrixVector(int m, int origin, int bound) {
        // random number generator
        Random rd = new Random();

        // fill matrix and vector with random short values
        matrix = new short[m * m];
        vector = new short[m];

        for (int i = 0; i < m; i++) {
            vector[i] = (short) (rd.nextInt(bound - origin) + origin);
            for (int j = 0; j < m; j++) {
                matrix[i * m + j] = (short) (rd.nextInt(bound - origin) + origin);
            }
        }

        this.m = m;
    }

    /**
     * Compute the matrix-vector product sequentially.
     *
     * @return the matrix-vector product
     */
    public short[] sequential() {
        solution = new short[this.m];

        // compute matrix-vector product
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                solution[i] += matrix[i * m + j] * vector[j];//matrix[i][j] * vector[j];
            }
        }

        return solution;
    }

    /* ###############################################################*/
    /* #################   PARALLEL PROCEDURES       #################*/
    /* ###############################################################*/

    short[] streamSolution;
    private boolean parallelInitialized = false;
    private boolean parallelRun = false;
    private long[] local_work_size;
    private long[] global_work_size;

    /**
     * Perform the computation of the matrix-vector product using a parallel stream.
     *
     * @return the matrix-vector product
     */
    public short[] parallelAsStream() {
        streamSolution = new short[this.m];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < m; j++) {
                streamSolution[i] += matrix[i * m + j] * vector[j];
            }
        });
        return streamSolution;
    }

    /**
     * Get the actual local_work_size. During setup it may have changed!
     */
    public long getLocalWorkSize(){
        if(local_work_size != null)
            return local_work_size[0];
        return -1;
    }

    /**
     * Initialize parallel procedures.
     *
     * @param lws the local work size
     */
    public void initParallel(long lws, int availableUnits)
    {
        if(!OpenCL.isInitialized())
            throw new RuntimeException("Initialize OpenCL first!");

        // create the kernel to run parallel matrix multiplication
        // use file if it is present
        // if file not present maybe in jar file?
        try {
            var kernelName = "matrix_vec";
            var kernelFile = "%s.cl".formatted(kernelName);
            var filePath = Path.of(kernelFile);

            if(Files.exists(filePath))
                OpenCL.createKernel(Files.readString(filePath), kernelName);
            else
            {
                // try jar resource file
                try (var inputStream = this.getClass().getClassLoader().getResourceAsStream(kernelFile)) {
                    var result = new BufferedReader(new InputStreamReader(inputStream))
                            .lines().parallel().collect(Collectors.joining("\r\n"));
                    OpenCL.createKernel(result, kernelName);
                }
            }





        } catch (IOException e) {
            throw new RuntimeException("Kernel file was not found!");
        }

        memBuffers = new cl_mem[3];
        memBuffers[0] = clCreateBuffer(OpenCL.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_short * m * m, Pointer.to(matrix), null);
        memBuffers[1] = clCreateBuffer(OpenCL.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_short * m, Pointer.to(vector), null);
        memBuffers[2] = clCreateBuffer(OpenCL.context,
                CL_MEM_READ_WRITE,
                (long) Sizeof.cl_short * m, null, null);

        int[] ms = new int[]{m};
        // specify kernel arguments to be passed into the kernels
        clSetKernelArg(OpenCL.kernel, 0,
                Sizeof.cl_mem, Pointer.to(memBuffers[0]));
        clSetKernelArg(OpenCL.kernel, 1,
                Sizeof.cl_mem, Pointer.to(memBuffers[1]));
        clSetKernelArg(OpenCL.kernel, 2,
                Sizeof.cl_mem, Pointer.to(memBuffers[2]));
        clSetKernelArg(OpenCL.kernel, 3,
                Sizeof.cl_int, Pointer.to(ms));

        // global work size is just the size of the matrix
        global_work_size = new long[]{m};

        // lws = -1 -> automatic local_work_size
        if(lws == -2){
            var computeWorkGroups = (double)m / ((double)availableUnits * 2.0);
            var computeLWS = m/(long)Math.ceil(computeWorkGroups);
            local_work_size = new long[]{computeLWS};
            local_work_size[0] = Math.min(local_work_size[0], global_work_size[0]);
        }else if(lws == -1) {
            local_work_size = null;
        } else {
            local_work_size = new long[]{lws};
            // this has to be done in case the supplied work size is bigger than global work size
            local_work_size[0] = Math.min(local_work_size[0], global_work_size[0]);
        }

        parallelInitialized = true;
    }

    /**
     * Read parallel results from commandQueue.
     *
     * @return Results in array.
     */
    public short[] readParallel()
    {
        if(!parallelRun)
            throw new RuntimeException("parallel() must be called before reading results!");

        // read result into buffer
        short [] result = new short[this.m];
        clEnqueueReadBuffer(OpenCL.commandQueue, memBuffers[2], CL_TRUE, 0,
                (long) m * Sizeof.cl_short, Pointer.to(result), 0, null, null);

        // wait for commandqueue to finish
        clFinish(OpenCL.commandQueue);

        return result;
    }

    /**
     * Execute matrix multiplication in parallel using openCL.
     *
     */
    public void parallel()
    {
        if(!parallelInitialized || !OpenCL.hasKernel())
            throw new RuntimeException("Parallel core and OpenCL must be initialized first!");

        // this only queues the kernels for execution
        // they do not run either clFinish is called or the output buffer is read
        clEnqueueNDRangeKernel(OpenCL.commandQueue, OpenCL.kernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        // wait for commandqueue to finish
        clFinish(OpenCL.commandQueue);

        parallelRun = true;
    }


    public void releaseParallel(){
        for(cl_mem buff : memBuffers){
            clReleaseMemObject(buff);
        }
    }

    public int getMatrixSize() {
        return m;
    }
}
