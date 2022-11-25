package task1;

import org.jocl.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.jocl.CL.*;
import static org.jocl.CL.clSetKernelArg;

/**
 * Class wrapping a matrix lhs and a vector rhs.
 */
public class MatrixVector {
    private cl_program program;
    private cl_kernel kernel;

    // create memory objects
    private cl_mem[] memBuffers;

    int m;
    // left hand side
    double[][] matrix;
    // right hand side
    double[] vector;

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

        // fill matrix and vector with random double values
        double[][] matrix = new double[m][m];
        double[] vector = new double[m];
        for (int i = 0; i < m; i++) {
            vector[i] = (rd.nextInt(bound - origin) + origin)*rd.nextDouble();
            for (int j = 0; j < m; j++) {
                matrix[i][j] = (rd.nextInt(bound - origin) + origin)*rd.nextDouble();
            }
        }

        this.m = m;
        this.matrix = matrix;
        this.vector = vector;
    }

    /**
     * Compute the matrix-vector product sequentially.
     *
     * @return the matrix-vector product
     */
    public double[] sequential() {
        double [] result = new double[this.m];

        // compute matrix-vector product
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        return result;
    }

    /* ###############################################################*/
    /* #################   PARALLEL PROCEDURES       #################*/
    /* ###############################################################*/

    private boolean parallelInitialized = false;
    private boolean parallelRun = false;

    /**
     * Initialize parallel procedures.
     *
     * @param context   OpenCL Context.
     */
    public void initParallel(cl_context context)
    {
        // create the kernel to run parallel matrix multiplication
        try {
            createKernel(context);
        } catch (IOException e) {
            throw new RuntimeException("Kernel file was not found!");
        }

        // concatenate values
        int n = m * m;
        double[] values = new double[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                int id = m * i + j;
                values[id] = matrix[i][j];
            }
        }

        memBuffers = new cl_mem[3];
        memBuffers[0] = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_double * n, Pointer.to(values), null);
        memBuffers[1] = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_double * m, Pointer.to(vector), null);
        memBuffers[2] = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                (long) Sizeof.cl_double * m, null, null);

        // specify kernel arguments to be passed into the kernels
        clSetKernelArg(kernel, 0,
                Sizeof.cl_mem, Pointer.to(memBuffers[0]));
        clSetKernelArg(kernel, 1,
                Sizeof.cl_mem, Pointer.to(memBuffers[1]));
        clSetKernelArg(kernel, 2,
                Sizeof.cl_mem, Pointer.to(memBuffers[2]));
        clSetKernelArg(kernel, 3,
                Sizeof.cl_int, Pointer.to(new int[]{m}));

        parallelInitialized = true;
    }

    /**
     * Read parallel results from commandQueue.
     *
     * @param commandQueue  The queue to read from.
     * @return Results in array.
     */
    public double[] readParallel(cl_command_queue commandQueue)
    {
        if(!parallelRun)
            throw new RuntimeException("parallel(commandQueue) must be called before reading results!");

        // read result into buffer
        double [] result = new double[this.m];
        clEnqueueReadBuffer(commandQueue, memBuffers[2], CL_TRUE, 0,
                (long) m * Sizeof.cl_double, Pointer.to(result), 0, null, null);
        return result;
    }

    /**
     * Execute matrix multiplication in parallel using openCL.
     *
     * @param commandQueue CommandQueue of the context.
     */
    public void parallel(cl_command_queue commandQueue)
    {
        if(!parallelInitialized)
            throw new RuntimeException("Parallel core must be initialized first using init_parallel()!");

        long[] global_work_size = new long[]{m};
        long[] local_work_size = new long[]{1};

        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        parallelRun = true;
    }

    /**
     * Create kernel for the parallel execution on GPU context.
     *
     * @param context: The context the kernel should run in.
     * @throws IOException: kernel.cl not found.
     */
    private void createKernel(cl_context context) throws IOException {
        String kernelSource = Files.readString(Path.of("kernel.cl"));

        // create the program from the source code
        program = clCreateProgramWithSource(context,
                1, new String[]{ kernelSource }, null, null);

        // build the program
        clBuildProgram(program, 0, null, null, null, null);

        // create the kernel
        kernel = clCreateKernel(program, "matrix_vec", null);
    }

}
