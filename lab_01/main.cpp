#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include <CL/cl.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main()
{
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
		std::ifstream cl_file("convolution.cl");
		std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

		// create program
		cl::Program program(context, source);

		// compile opencl source
		try
		{
			program.build(devices, "-DBLOCK_SIZE=256");
		}
		catch (cl::Error const & e)
		{
			std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
			std::cout << log_str;
			return 0;
		}

		int n, m;
		std::ifstream input("input.txt");
		input >> n >> m;

		std::vector<float> a(n * n);
		std::vector<float> b(m * m);

		for (int i = 0; i < n * n; i++) {
			int x;
			input >> x;
			a[i] = x;
		}

		for (int i = 0; i < m * m; i++) {
			int x;
			input >> x;
			b[i] = x;
		}
		input.close();

		std::vector<float> ans(n * n, 0);
		// allocate device buffer to hold message
		cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * a.size());
		cl::Buffer dev_b(context, CL_MEM_WRITE_ONLY, sizeof(float) * b.size());
		cl::Buffer dev_ans(context, CL_MEM_WRITE_ONLY, sizeof(float) * ans.size());

		auto const& kernel = "gpu_convolution";

		// copy from cpu to gpu
		queue.enqueueWriteBuffer(dev_a, CL_FALSE, 0, sizeof(float) * a.size(), &a[0]);
		queue.enqueueWriteBuffer(dev_b, CL_FALSE, 0, sizeof(float) * b.size(), &b[0]);

		// load named kernel from opencl source
		cl::Kernel kernel_gmem(program, kernel);
		// Make kernel can be used here
		kernel_gmem.setArg(0, dev_a);
		kernel_gmem.setArg(1, dev_b);
		kernel_gmem.setArg(2, dev_ans);
		kernel_gmem.setArg(3, n);
		kernel_gmem.setArg(4, m);

		size_t const block_size = 16;
		size_t thread_size = block_size;
		while (thread_size < n) 
			thread_size *= 2;
		std::cout << thread_size << ' ';
		cl::Event event;
		queue.enqueueNDRangeKernel(kernel_gmem,
			cl::NullRange,
			cl::NDRange(thread_size, thread_size),
			cl::NDRange(block_size, block_size),
			nullptr,
			&event);

		event.wait();
		queue.enqueueReadBuffer(dev_ans, CL_TRUE, 0, sizeof(float) * ans.size(), &ans[0]);
		std::ofstream output("output.txt");
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				output << ans[i * n + j] << " ";
			output << '\n';
		}
		output.close();
	}
	catch (cl::Error const & e)
	{
		std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
	}

	return 0;
}