__kernel void gpu_convolution(__global float * a, __global float * b, __global float * ans, int n, int m)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	if (i >= n || j >= n)
		return;

	float value = 0;
	int hm = (m - 1) / 2;
	for (int s = -hm; s <= hm; s++)
	{
		if (i + s < 0 || i + s >= n)
			continue;
		for (int r = -hm; r <= hm; r++)
		{
			if (j + r < 0 || j + r >= n)
				continue;
			value += a[(i + s) * n + j + r] * b[(s + hm) * m + r + hm];
		}
	}
	ans[i * n + j] = value;
	barrier(CLK_GLOBAL_MEM_FENCE);
}