#include <iostream>
#include "CL/cl.hpp"
#include <vector>
#include <complex>
#include <algorithm>

int main() {

    cl::Device device;     //No Multi-GPU Support in this proof-of-concept
    cl::Platform platform; //No Multi-GPU Support in this proof-of-concept
    int returnCode = 0;

    //Find an OpenCL device
    {
        std::vector<cl::Platform> all_platforms;
        std::vector<cl::Device> all_devices;
        bool choseDevice = false;
        returnCode = cl::Platform::get(&all_platforms);
        if (returnCode != 0) { //cl::Platform::get returned a non-zero return code, there was an error
            std::cout << "Error (OpenCL Error No. " << returnCode << ") when fetching platforms, exiting now..." << std::endl;
            return -1;
        }
        if (all_platforms.size() == 0) { //cl::Platform::get returned no OpenCL Platforms
            std::cout << "Error: No Platforms were found. Exiting now..." << std::endl;
            return -1;
        }
        for (auto& plat : all_platforms) { //Loop through all found platforms
            all_devices.resize(0); //Reset vector
            returnCode = plat.getDevices(CL_DEVICE_TYPE_GPU, &all_devices); //Fetch GPUs an Accelerators

            if (all_devices.size() == 0) { //No devices matching criteria in this platform
                continue; //Continue with next platform
            }
            platform = plat;         //Select the current platform

            std::string platformName;
            returnCode = platform.getInfo(CL_PLATFORM_NAME, &platformName);
            if (returnCode != 0) { //There was an error while retrieving the platform name
                std::cout << "Error: Successfully selected platform, but it was not possible to get its name (OpenCL Error No. " << returnCode << "). Exiting now..." << std::endl;
                return -1;
            }
            std::cout << "Info: Successfully selected platform ('" << platformName << "')" << std::endl;


            std::string deviceName;
            device = all_devices[0]; //Select the first available device
            returnCode = device.getInfo(CL_DEVICE_NAME, &deviceName);
            if (returnCode != 0) {
                std::cout << "Error: Successfully selected device, but it was not possible to get its name (OpenCL Error No. " << returnCode << "). Exiting now..." << std::endl;
                return -1;
            }
            std::cout << "Info: Successfully selected device ('" << deviceName << "')" << std::endl;

            choseDevice = true;      //Indicate that we successfully selected a device
            break;                   //No need to continue, we selected a device
        }
        if (!choseDevice) {
            std::cout << "Error: No devices (GPUs) were found, exiting now..." << std::endl;
            return -1;
        }
    }

    cl::Context context = cl::Context{ device, NULL, NULL, NULL, &returnCode };
    if (returnCode != 0) {
        std::cout << "Error when initializing cl::Context for chosen device and platform (OpenCL Error No. " << returnCode << "). Exiting now..." << std::endl;
        return -1;
    }
    std::cout << "Info: Initialized cl::Context for chosen device and platform" << std::endl;

    cl::CommandQueue queue{ context, device, 0U, &returnCode }; //Create command queue
    if (returnCode != 0) {
        std::cout << "Error when creating command queue (OpenCL Error No. " << returnCode << "). Exiting now..." << std::endl;
        return -1;
    }
    std::cout << "Info: Initialized cl::CommandQueue for chosen device and platform" << std::endl;

	//CRICITAL VALUE: 23170! 23171 is the first non-working value
    constexpr size_t targetMatrixSize = 23170; //arbitrary value, increase if bug does not occur

	static_assert(targetMatrixSize >= 1000, "Matrix must be big! 1000 is not enough...");

    //The maximum allocatable bytes per buffer
    size_t maxBufferBytes = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(&returnCode);
    if (returnCode != 0) {
        std::cout << "Error when retrieving CL_DEVICE_MAX_MEM_ALLOC_SIZE (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
        return -1;
    }

    //The maximum bytes the device memory can hold
    size_t maxGlobalBytes = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&returnCode);
    if (returnCode != 0) {
        std::cout << "Error when retrieving CL_DEVICE_GLOBAL_MEM_SIZE (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
        return -1;
    }

    //Maximum size of the matrix stripe
    size_t maxMatrixStripeSize = std::min<size_t>(maxBufferBytes, maxGlobalBytes);

    //maximum matrix size per buffer, solve matrixSizeInBytes = n² * sizeof(std::complex<float>) for n
    size_t matrixSizeLimitedByMaxBufferBytes = static_cast<size_t>(std::floor(std::sqrt((double)(maxBufferBytes) / (double) sizeof(std::complex<float>))));

    //maximum matrix size limited by global memory, solve n² = sizeof(std::complex<float>) for n
    size_t matrixSizeLimitedByMaxGlobalBytes = static_cast<size_t>(std::floor(std::sqrt((double)(maxGlobalBytes) / (double) sizeof(std::complex<float>))));

    size_t matrixSizeLimit = std::min<size_t>(matrixSizeLimitedByMaxBufferBytes, matrixSizeLimitedByMaxGlobalBytes);

    std::cout << "Info: Targeted matrix size: " << targetMatrixSize << std::endl;
    std::cout << "Info: Matrix size limit: " << matrixSizeLimit << std::endl;

	size_t columnLimit = static_cast<size_t>(std::floor((double)maxMatrixStripeSize / ((double)targetMatrixSize * (double) sizeof(std::complex<float>))));

    if (targetMatrixSize <= matrixSizeLimit) {
        std::cout << "Info: Targeted matrix size is too small, bug will not occur, because matrix fits in memory" << std::endl;
		std::cout << "Info: Setting columnLimit to an artificially small value" << std::endl;
		columnLimit = static_cast<size_t>(std::floor((double)targetMatrixSize / 2.0));
    } else {
        std::cout << "Info: Targeted matrix size is big enough, matrix will be split up into stripes" << std::endl;
    }

    

    if (columnLimit == 0) {
        std::cout << "Error: Matrix size is too big. Device can not find at least one column of the matrix" << std::endl;
        return -1;
    }
    if (columnLimit < 500) {
        std::cout << "Warning: The number of columns the matrix can fit is very small!" << std::endl;
    }

    unsigned int completeMatrixStripes = static_cast<unsigned int>(std::floor((double) targetMatrixSize / (double) columnLimit));

    size_t remainingColumns = targetMatrixSize - completeMatrixStripes * columnLimit;

    bool hasRemainingColumns = remainingColumns > 0;

    if (hasRemainingColumns) {
        std::cout << "Info: Matrix is split into " << completeMatrixStripes << " stripes with " << columnLimit << " columns and one pass for the " << remainingColumns << " remaining columns" << std::endl;
    } else {
        std::cout << "Info: Matrix is split into " << completeMatrixStripes << " stripes with " << columnLimit << " columns, no extra pass is necessary" << std::endl;
    }

    size_t numStripeElements = columnLimit * targetMatrixSize;
    size_t deviceBufferSize = numStripeElements * sizeof(std::complex<float>);

    std::complex<float>* hostMemory = new std::complex<float>[targetMatrixSize * targetMatrixSize];

    //Allocate buffer for first matrix to maximum possible size
    cl::Buffer workspaceBuffer { context, CL_MEM_WRITE_ONLY, deviceBufferSize, NULL, &returnCode };
    if (returnCode != 0) {
        std::cout << "Could not create workspacebuffer on device (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
        return -1;
    }

    //Read offset in the buffer, which is always zero
    cl::size_t<3> buffer_offset;
    buffer_offset[0] = 0;
    buffer_offset[1] = 0;
    buffer_offset[2] = 0;

    //Write offset in the host, only the first component will be changed in every iteration
    cl::size_t<3> host_offset;
    host_offset[0] = 0;
    host_offset[1] = 0;
    host_offset[2] = 0;

    //Region to read and write
    cl::size_t<3> region;
    region[0] = columnLimit * sizeof(std::complex<float>); //Bytes per row
    region[1] = targetMatrixSize; //Length of column
    region[2] = 1; //1 Due to 2D Copy

    size_t buffer_row_pitch = columnLimit * sizeof(std::complex<float>); //Buffer has columnLimit columns width
    size_t buffer_slice_pitch = 0;

    size_t host_row_pitch = targetMatrixSize * sizeof(std::complex<float>); //Host has targetMeshSize columns width
    size_t host_slice_pitch = 0;

	std::cout << "Running without profiling and not blocking (for CodeXL)" << std::endl;

    for (unsigned int fullEnqueueCounter = 0; fullEnqueueCounter < completeMatrixStripes; ++fullEnqueueCounter) {

        unsigned int firstColumnIndexOfIteration = fullEnqueueCounter * columnLimit; //The index of the leftmost column of the current part

        host_offset[0] = firstColumnIndexOfIteration * sizeof(std::complex<float>);

        std::cout << "Attempting to read buffer no." << fullEnqueueCounter + 1 << std::endl;
        //Read results back
        returnCode = queue.enqueueReadBufferRect(workspaceBuffer, CL_FALSE, buffer_offset, host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, hostMemory);
        if (returnCode != 0) {
            std::cout << "Could not enqueue rectRead workspaceBuffer (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
            return -1;
        }
        std::cout << "Read Buffer!" << std::endl;
		queue.flush();
		queue.finish();

    }

    if (hasRemainingColumns) {

        unsigned int firstColumnIndexOfIteration = completeMatrixStripes * columnLimit; //The index of the leftmost column of the current part

        host_offset[0] = firstColumnIndexOfIteration * sizeof(std::complex<float>); //offset in host buffer, as before

        //Note: This is the last step, we just process remainingIndexes columns
        region[0] = remainingColumns * sizeof(std::complex<float>); //Bytes per remaining row

        std::cout << "Attempting to last buffer" << std::endl;
        //Read results back
        returnCode = queue.enqueueReadBufferRect(workspaceBuffer, CL_FALSE, buffer_offset, host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, hostMemory);
        if (returnCode != 0) {
            std::cout << "Could not enqueue rectRead workspaceBuffer (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
            return -1;
        }
        std::cout << "Read Buffer!" << std::endl;
		queue.finish();
		queue.flush();
    }

	std::cout << "All Buffers were successfully read" << std::endl;

	std::cout << "Now running with profiling, not blocking" << std::endl;
	region[0] = columnLimit * sizeof(std::complex<float>); //Bytes per row

	queue = cl::CommandQueue { context, device, CL_QUEUE_PROFILING_ENABLE, &returnCode }; //Create new command queue
	if (returnCode != 0) {
		std::cout << "Error when recreating command queue (OpenCL Error No. " << returnCode << "). Exiting now..." << std::endl;
		return -1;
	}
	std::cout << "Recreated Command Queue" << std::endl;

	cl::Event profilingEvent;

	for (unsigned int fullEnqueueCounter = 0; fullEnqueueCounter < completeMatrixStripes; ++fullEnqueueCounter) {

		unsigned int firstColumnIndexOfIteration = fullEnqueueCounter * columnLimit; //The index of the leftmost column of the current part

		host_offset[0] = firstColumnIndexOfIteration * sizeof(std::complex<float>);

		std::cout << "Attempting to read buffer no." << fullEnqueueCounter + 1 << std::endl;
		//Read results back
		returnCode = queue.enqueueReadBufferRect(workspaceBuffer, CL_FALSE, buffer_offset, host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, hostMemory, NULL, &profilingEvent);
		if (returnCode != 0) {
			std::cout << "Could not enqueue rectRead workspaceBuffer (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
			return -1;
		}
		std::cout << "Read Buffer!" << std::endl;
		queue.flush();
		queue.finish();

		cl_ulong start = profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>(&returnCode);
		if (returnCode != 0) {
			std::cout << "Could not get CL_PROFILING_COMMAND_START" << std::endl;
			return -1;
		}

		cl_ulong end = profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>(&returnCode);
		if (returnCode != 0) {
			std::cout << "Could not get CL_PROFILING_COMMAND_END" << std::endl;
			return -1;
		}

		cl_ulong time_diff = end - start;

		std::cout << "Data transfer took " << time_diff << "ns, start at " << start << ", end at " << end << std::endl;

	}

	if (hasRemainingColumns) {

		unsigned int firstColumnIndexOfIteration = completeMatrixStripes * columnLimit; //The index of the leftmost column of the current part

		host_offset[0] = firstColumnIndexOfIteration * sizeof(std::complex<float>); //offset in host buffer, as before

																					//Note: This is the last step, we just process remainingIndexes columns
		region[0] = remainingColumns * sizeof(std::complex<float>); //Bytes per remaining row

		std::cout << "Attempting to last buffer" << std::endl;
		//Read results back
		returnCode = queue.enqueueReadBufferRect(workspaceBuffer, CL_FALSE, buffer_offset, host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, hostMemory, NULL, &profilingEvent);
		if (returnCode != 0) {
			std::cout << "Could not enqueue rectRead workspaceBuffer (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
			return -1;
		}
		std::cout << "Read Buffer!" << std::endl;
		queue.finish();
		queue.flush();
		cl_ulong start = profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>(&returnCode);
		if (returnCode != 0) {
			std::cout << "Could not get CL_PROFILING_COMMAND_START" << std::endl;
			return -1;
		}

		cl_ulong end = profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>(&returnCode);
		if (returnCode != 0) {
			std::cout << "Could not get CL_PROFILING_COMMAND_END" << std::endl;
			return -1;
		}

		cl_ulong time_diff = end - start;

		std::cout << "Data transfer took " << time_diff << "ns, start at " << start << ", end at " << end << std::endl;
	}

	std::cout << "Now running with profiling AND blocking (CL_TRUE)" << std::endl;
	region[0] = columnLimit * sizeof(std::complex<float>); //Bytes per row
	for (unsigned int fullEnqueueCounter = 0; fullEnqueueCounter < completeMatrixStripes; ++fullEnqueueCounter) {

		unsigned int firstColumnIndexOfIteration = fullEnqueueCounter * columnLimit; //The index of the leftmost column of the current part

		host_offset[0] = firstColumnIndexOfIteration * sizeof(std::complex<float>);

		std::cout << "Attempting to read buffer no." << fullEnqueueCounter + 1 << std::endl;
		//Read results back
		returnCode = queue.enqueueReadBufferRect(workspaceBuffer, CL_TRUE, buffer_offset, host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, hostMemory, NULL, &profilingEvent);
		if (returnCode != 0) {
			std::cout << "Could not enqueue rectRead workspaceBuffer (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
			return -1;
		}
		std::cout << "Read Buffer!" << std::endl;

		cl_ulong start = profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>(&returnCode);
		if (returnCode != 0) {
			std::cout << "Could not get CL_PROFILING_COMMAND_START" << std::endl;
			return -1;
		}

		cl_ulong end = profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>(&returnCode);
		if (returnCode != 0) {
			std::cout << "Could not get CL_PROFILING_COMMAND_END" << std::endl;
			return -1;
		}

		cl_ulong time_diff = end - start;

		std::cout << "Data transfer took " << time_diff << "ns, start at " << start << ", end at " << end << std::endl;

	}

	if (hasRemainingColumns) {

		unsigned int firstColumnIndexOfIteration = completeMatrixStripes * columnLimit; //The index of the leftmost column of the current part

		host_offset[0] = firstColumnIndexOfIteration * sizeof(std::complex<float>); //offset in host buffer, as before

																					//Note: This is the last step, we just process remainingIndexes columns
		region[0] = remainingColumns * sizeof(std::complex<float>); //Bytes per remaining row

		std::cout << "Attempting to last buffer" << std::endl;
		//Read results back
		returnCode = queue.enqueueReadBufferRect(workspaceBuffer, CL_TRUE, buffer_offset, host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, hostMemory, NULL, &profilingEvent);
		if (returnCode != 0) {
			std::cout << "Could not enqueue rectRead workspaceBuffer (OpenCL Error No. " << returnCode << "), exiting now..." << std::endl;
			return -1;
		}
		std::cout << "Read Buffer!" << std::endl;

		cl_ulong start = profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>(&returnCode);
		if (returnCode != 0) {
			std::cout << "Could not get CL_PROFILING_COMMAND_START" << std::endl;
			return -1;
		}

		cl_ulong end = profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>(&returnCode);
		if (returnCode != 0) {
			std::cout << "Could not get CL_PROFILING_COMMAND_END" << std::endl;
			return -1;
		}

		cl_ulong time_diff = end - start;

		std::cout << "Data transfer took " << time_diff << "ns, start at " << start << ", end at " << end << std::endl;
	}



    return 0;
}