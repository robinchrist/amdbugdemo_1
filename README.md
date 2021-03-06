With the current AMD Radeon 18.3.4 drivers, there seems to be an issue using clEnqueueReadBufferRect (I haven't tested clEnqueueWriteBufferRect though) with large numbers:

Suppose we want to generate BEM-matrices on the GPU. For problem sizes above a specific limit, the problem cannot be calculated in one run because either the total available memory size on the device or the maximum available memory per buffer is too small to hold the full matrices.

We now split the problem and just calculate parts of the matrices on the GPU.

Suppose our problem is large, say n=23171.

The host buffer now has a row width (for std::complex<float>) of 23171 * 8 = 185368 bytes.

The kernel now generates a certain number of columns of the whole matric, hence clEnqueueReadBufferRect.

Because our matrix is quadratic, the length of the rows, region[1], is 23171.

Even though nothing is special about the number 23171 it is not possible to profile the clEnqueueReadBufferRect workloads anymore, because the values getProfilingInfo<CL_PROFILING_COMMAND_END>() and CL_PROFILING_COMMAND_STARTreturned from the profiling events are identical!

However, let's take a look at the parameters of clEnqueueReadBufferRect:

I identified region[1] and host_row_pitch as the offending parameters.

A few examples might be (first the region[1] parameter and afterwards the how_row_pitch parameter):


23171 | 23171 * 8 -> FAIL


23170 | 23171 * 8  -> FAIL


23170 | (23171 * 8) - 1 = 23170 | 185367 -> WORKS

Why does 23170 | 185367 work? Well, let's take a look at the products of the parameters:


23171 * 23171 * 8 = 23171 * 185368 = 4 295 161 928 -> FAIL


23170 * 23171 * 8 = 23170 * 185368 = 4 294 976 560 -> FAIL

23170 * 185367 = 4 294 953 390 -> WORKS

Note the limit of uint32_t: 4 294 967 295! 

# TL; DR
For targetMatrixSize <= 23170 profiling is working, for targetMatrixSize > 23171 profiling does not work
CL_PROFILING_COMMAND_START == CL_PROFILING_COMMAND_END

There are 3 passes:
1. Without profiling and not blocking, to show the correct functionality of CodeXL
2. With profiling, not blocking to extract the values of the profiling events
3. With profiling AND blocking to show that CodeXL is able to track the duration of the transfer is they are blocking (CL_TRUE), however CodeXL calculates the runtime by using the values provided by the profiling events

Furthermore, the transfers are MUCH slower for n > 23170!
