// uncomment pragma and printf to show debug information
//#pragma OPENCL EXTENSION cl_intel_printf : enable

__kernel void matrix_vec(__global short *lhs, __global short *rhs, __global short *result, const int d) {
    int gid = get_global_id(0);
    int size = d;
    result[gid] = 0;
    // printf("gid: %d, size: %d, lhs[0]: %d, rhs[0]: %d\n", gid, d, lhs[0], rhs[0]);
    for(int i = 0; i < size; ++i) {
        result[gid] += lhs[gid * size + i] * rhs[i];
    }
    // printf("%d: %d\n", gid, result[gid]);
};