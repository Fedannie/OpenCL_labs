__kernel void gpu_prefix(__global float * a, __global float * extra, __global float * ans, __local float * r, __local float * s, int n) {
	size_t block_size = get_local_size(0);
	size_t threads_size = get_global_size(0);
	size_t g_id = get_global_id(0);
	size_t l_id = get_local_id(0);
	
	extra[g_id] = a[g_id];
	barrier(CLK_GLOBAL_MEM_FENCE);

	int shift = 0;
	int extra_shift = 0;
	size_t step = 1;

	while(step < n)
	{
		r[shift + l_id] = extra[extra_shift + g_id]; 
		s[shift + l_id] = extra[extra_shift + g_id];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (size_t k = 1; k < block_size; k <<= 1)
		{
			if (step * g_id < n)
				s[shift + l_id] = r[shift + l_id] + (l_id > k - 1 ? r[shift + l_id - k] : 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			__local float * tmp = r;
			r = s;
			s = tmp;
		}

		extra_shift += threads_size / step;

		if ((l_id == 0) && (step * (g_id + block_size - 1) < n))
			extra[extra_shift + g_id / block_size] = r[shift + block_size - 1];
		barrier(CLK_GLOBAL_MEM_FENCE);

		shift += block_size;
		step *= block_size;
	}

	while (step > 0)
	{
		shift -= block_size;
		if ((g_id >= block_size) && (step * g_id / block_size < n))
			r[shift + l_id] += extra[extra_shift + g_id / block_size - 1];
		extra_shift -= threads_size / step;
		if (step * g_id < n)
			extra[extra_shift + g_id] = r[shift + l_id];
		step /= block_size;
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	ans[g_id] = r[l_id];
}