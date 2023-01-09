__kernel void matrix_vector_multiplication(__global float *a, __global float *b, __global float *c, const int d) {
    int gid = get_global_id(0);
    int size = d;
    c[gid] = 0;
    for(int i = 0; i < size; ++i) {
        c[gid] += a[gid * size + i] * b[i];
    }
};