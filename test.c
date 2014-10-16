__kernel void x2(
        __global const uint *input,
        __global uint *output,
        uint width)
{
        uint x = get_global_id(0);
        uint y = get_global_id(1);
        uint idx = y*width + x;
        uint sum = input[idx];
        uint i;
        uint cnt = 0;
        uint mask = 1;
        for (i = 0; i < 32; i++) {
                if (sum & mask)
                        cnt++;
                mask = mask << 1;
        }
        output[idx] = cnt;
}
