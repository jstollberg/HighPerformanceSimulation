__kernel void matrix_vec(__global float *lhs, __global float *rhs, __global float *result, const int d) {
    int gid = get_global_id(0);
    int size = d;
    result[gid] = 0;
    for(int i = 0; i < size; ++i) {
        result[gid] += lhs[gid * size + i] * rhs[i];
    }
};