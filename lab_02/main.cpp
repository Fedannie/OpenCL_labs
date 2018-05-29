#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include <CL/cl.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main() {
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	std::vector<cl::Kernel> kernels;

	try {
		// create platform
		cl::Platform::get(&platforms);
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

		// create context
		cl::Context context(devices);

		// create command queue
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		// load opencl source
		std::ifstream cl_file("prefix.cl");
		std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
			cl_string.length() + 1));

		// create program
		cl::Program program(context, source);

		// compile opencl source
		try
		{
			program.build(devices, "-DBLOCK_SIZE=16");
		}
		catch (cl::Error const & e)
		{
			std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
			std::cout << log_str;
			return 0;
		}

		int n;
		std::ifstream input("input.txt");
		input >> n;
		std::vector<float> a(n);

		for (int i = 0; i < n; i++)
			input >> a[i];
		input.close();

		std::vector<float> ans(n, 0);
		std::vector<float> extra(2 * n, 0);

		// allocate device buffer to hold message
		cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * a.size());
		cl::Buffer dev_ans(context, CL_MEM_WRITE_ONLY, sizeof(float) * ans.size());
		cl::Buffer dev_extra(context, CL_MEM_READ_WRITE, sizeof(float) * extra.size());

		auto const& kernel = "gpu_prefix";

		// copy from cpu to gpu
		queue.enqueueWriteBuffer(dev_input, CL_FALSE, 0, sizeof(float) * a.size(), &a[0]);
		queue.enqueueWriteBuffer(dev_extra, CL_FALSE, 0, sizeof(float) * extra.size(), &extra[0]);

		size_t const block_size = 16;
		int thread_size = block_size;
		while (thread_size < n)
			thread_size *= 2;
		int needed_memory = 1;
		int _tmp = thread_size;
		while (_tmp > 0) {
			needed_memory++;
			_tmp /= block_size;
		}

		// load named kernel from opencl source
		cl::Kernel kernel_gmem(program, kernel);
		// Make kernel can be used here
		kernel_gmem.setArg(0, dev_input);
		kernel_gmem.setArg(1, dev_extra);
		kernel_gmem.setArg(2, dev_ans);
		kernel_gmem.setArg(3, cl::__local(sizeof(float) * needed_memory * block_size));
		kernel_gmem.setArg(4, cl::__local(sizeof(float) * needed_memory * block_size));
		kernel_gmem.setArg(5, n);

		cl::Event event;
		queue.enqueueNDRangeKernel(kernel_gmem,
			cl::NullRange,
			cl::NDRange(thread_size),
			cl::NDRange(block_size),
			nullptr,
			&event);
		event.wait();

		queue.enqueueReadBuffer(dev_ans, CL_TRUE, 0, sizeof(float) * ans.size(), &ans[0]);

		std::ofstream output("output.txt");
		for (int i = 0; i < n; i++)
			output << ans[i] << " ";
		output.close();
	}
	catch (cl::Error const & e) {
		std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
	}
	return 0;
}