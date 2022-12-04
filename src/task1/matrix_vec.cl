// uncomment pragma and printf to show debug information
//#pragma OPENCL EXTENSION cl_intel_printf : enable

// may use define to use constant values
// #define m {m}


__kernel void matrix_vec(__global const short *lhs, __global const short *rhs, __global short *result, const int d) {
    int gid = get_global_id(0);
    int size = d;
    result[gid] = 0;
    double sum = 0.0;
    // printf("gid: %d, size: %d, lhs[0]: %d, rhs[0]: %d\n", gid, d, lhs[0], rhs[0]);
    for(int i = 0; i < size; ++i) {
        sum += lhs[gid * size + i] * rhs[i];
    }
    // printf("%d: %d\n", gid, result[gid]);
    result[gid] = sum;
};