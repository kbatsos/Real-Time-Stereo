
typedef uint8_t uint8;
#define TB 128

#define DISP_MAX 256
#define BLOCK_SIZE 32
#define XDIM_MAX_THREADS 1024
#define BLOCK_D_SIZE 64

#define COLOR_DIFF(x, i, j) (abs(x[i] - x[j]))

struct postparams{
	float pi1;
    float pi2;
    float tau_so;
    float alpha1;
    float sgm_q1;
    float sgm_q2;
    float alpha2;
    float sigma; 
    int kernel_size; 
};

void parseConf(postparams &params,std::string conf ){
		const std::string& chars = "\t\n\v\f\r ";

	    std::ifstream ifs(conf.c_str());
        std::string line;
        if(ifs.is_open()){
                while(std::getline(ifs,line )){
                	   std::string opt = line.substr(0,line.find_last_of(":"));
                	   opt.erase(0, opt.find_first_not_of(chars));
                	   opt.erase(opt.find_last_not_of(chars) + 1);

                	   int start = line.find_last_of(":")+1;
                	   int end =  line.find_first_of("#") - start;

                	   std::string val = line.substr(start,end);

                	   val.erase(0, val.find_first_not_of(chars));
                	   val.erase(val.find_last_not_of(chars) + 1);

                	   if(!strcmp(opt.c_str(),"pi1")){
                	   		params.pi1 = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"pi2")){
                	   		params.pi2 = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"tau_so")){
                	   		params.tau_so = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"alpha1")){
                	   		params.alpha1 = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"sgm_q1")){
                	   		params.sgm_q1 = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"sgm_q2")){
                	   		params.sgm_q2 = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"alpha2")){
                	   		params.alpha2 = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"sigma")){
                	   		params.sigma = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"kernel_size")){
                	   		params.kernel_size = atoi(val.c_str());
                	   }  
                }

        }else{
                std::cout << "File " << conf << " does not exist! " <<std::endl;
                exit(0);
        }
}

std::vector<std::string> getImages(std::string file){
	    std::vector<std::string> imageNames;

        std::ifstream ifs(file.c_str());
        std::string line;
        if(ifs.is_open()){
                while(std::getline(ifs,line )){
                        imageNames.push_back(line);
                }

        }else{
                std::cout << "File " << file << " does not exist! " <<std::endl;
                exit(0);
        }

        return imageNames;

}

template<typename T>
__global__ void argmin( float* disp_d, T* cost, int rows, int cols, int ndisp ){


    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x; 

    if( Row < rows && Col < cols){

	    T mincost=cost[ Row*cols*ndisp+Col*ndisp ];
	    int d=0;
	    for(int i=1; i<ndisp; i++){
	    	float cd =  cost[ Row*cols*ndisp+Col*ndisp +i ];
	    	if( cd < mincost ){
	    		mincost = cd;
	    		d = i;
	    	}
	    }

	    disp_d[ Row*cols+Col ] = (float)d;

	}
}

template<typename T>
__global__ void argmin_d( float* disp_d, T* cost, int rows, int cols, int ndisp ){


    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x; 

    if( Row < rows && Col < cols){

	    T mincost=cost[ Row*cols+Col ];
	    int d=0;
	    for(int i=1; i<ndisp; i++){
	    	float cd =  cost[ i*rows*cols+Row*cols+Col ];
	    	if( cd < mincost ){
	    		mincost = cd;
	    		d = i;
	    	}
	    }

	    disp_d[ Row*cols+Col ] = (float)d;

	}
}

template<typename T>
__global__ void swap_axis(const T* __restrict__ cost, T* temp_cost, const int rows, const int cols, const int ndisp ){

	int Col = blockIdx.x*blockDim.x + threadIdx.x; 
    int Row = blockIdx.y*BLOCK_D_SIZE + threadIdx.y; 

    __shared__ T tile[BLOCK_D_SIZE][BLOCK_D_SIZE+1];



    if( Col< cols*rows){
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){
    		if(Row+d < ndisp)
    			tile[threadIdx.y+d][threadIdx.x] = cost [(Row+d)*rows*cols+Col ];
    	}	
    }


    	__syncthreads();

    	Col = blockIdx.x*blockDim.x+threadIdx.y;
    	Row = blockIdx.y*BLOCK_D_SIZE+threadIdx.x; 
    
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){
    		if((Col+d) < cols*rows && Row<ndisp)
    			temp_cost[ (Col+d)*ndisp+Row ] = tile[threadIdx.x][threadIdx.y+d];
	   	}
	    
	


}

template<typename T>
__global__ void swap_axis_back(const T* __restrict__ cost, T* temp_cost, const int rows, const int cols, const int ndisp ){

    int Col = blockIdx.x*blockDim.x + threadIdx.y;
    int Row = blockIdx.y*BLOCK_D_SIZE+threadIdx.x; 

    __shared__ T tile[BLOCK_D_SIZE][BLOCK_D_SIZE+1];



    if( Col< cols*rows){
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){

    		tile[threadIdx.y+d][threadIdx.x] = cost [(Col+d)*ndisp+Row  ];
    	}
    }


    	__syncthreads();


    	Col = blockIdx.x*blockDim.x + threadIdx.x; 
    	Row = blockIdx.y*BLOCK_D_SIZE + threadIdx.y; 
    
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){
    		if((Col+d) < cols*rows)
    			temp_cost[ (Row+d)*rows*cols+Col ] = tile[threadIdx.x][threadIdx.y+d];
	   	}
	    
	


}


template<typename T>
__global__ void transpose(const T* __restrict__ cost, T* temp_cost, const int dim1, const int dim2){

	
	int Col = blockIdx.x*blockDim.x + threadIdx.x; 
    int Row = blockIdx.y*BLOCK_D_SIZE + threadIdx.y; 
    int disp = blockIdx.z*dim1*dim2;

    __shared__ T tile[BLOCK_D_SIZE][BLOCK_D_SIZE+1];



    if( Col< dim2){
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){

    		if((Row+d)<dim1)
    			tile[threadIdx.y+d][threadIdx.x] = cost [disp+(Row+d)*dim2+Col ];
    	}	
    }


    	__syncthreads();

    	Col = blockIdx.x*blockDim.x+threadIdx.y;
    	Row = blockIdx.y*BLOCK_D_SIZE+threadIdx.x; 
    
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){
    		if((Col+d) < dim2 && Row < dim1)
    			temp_cost[disp+(Col+d)*dim1+Row ] = tile[threadIdx.x][threadIdx.y+d];
	   	}
	    
	


}

template<typename T>
__global__ void VerticalIntegralKernel(T* output, const int rows , const int cols , const int ndisp,const int offset){

    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* slice_sm = reinterpret_cast<T *>(shared_mem);    
    
    int Row = threadIdx.x+offset; 
    int Col = blockIdx.y; 
    int disp = blockIdx.z; 

    T val=0,temp=0,temp1=0;

    if( threadIdx.x < rows){ 
        val = output[disp*rows*cols+Row*cols+Col];
    

	    for(int i=1; i<32; i<<=1 ){
	    	temp = __shfl_up(val,i);
	    	if(  (threadIdx.x & 31)  >=i )
	    		val +=temp;

	    }

	    if( (threadIdx.x &  31) ==31 || threadIdx.x==(rows-1) )
	    	slice_sm[threadIdx.x/32] = val;

    }

    __syncthreads();

    temp=0;

    if( threadIdx.x < 32 ){
    	temp = slice_sm[threadIdx.x];

    	for(int i=1; i<32; i<<=1){
    		temp1 = __shfl_up(temp,i);
    		if( (threadIdx.x & 31)   >=i )
    			temp += temp1;
    	}

    	slice_sm[threadIdx.x] = temp;

    }

	__syncthreads();

	if( Row < rows){ 
		if(threadIdx.x >=32)
			val += slice_sm[threadIdx.x/32-1];
	
		output[disp*rows*cols+Row*cols+Col] = val;
	}

  
          
}

// This kernel has to be converted to the inplace integral kernel. Profiling though shows that this is not a bottleck for the method.
template<typename T,typename I>
__global__ void HorizontalIntegralKernel_outofplace(T* integral_vol,const I* input, const int integrrows , const int integrcols , const int ndisp,const int offset){
   
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* slice_sm = reinterpret_cast<T *>(shared_mem);    

    int Col = threadIdx.x+offset;
    int Row = blockIdx.x;
    int disp = blockIdx.z; 


    if( Col < integrcols && disp < ndisp){ 
        slice_sm[threadIdx.x] = input[disp*integrrows*integrcols+Row*integrcols+Col];
    }

    if(offset>0 & threadIdx.x==0){
    	slice_sm[threadIdx.x] = integral_vol[disp*integrrows*integrcols+Row*integrcols+Col];
    }    


    T sum;

        for(int stride=1; stride< blockDim.x; stride*=2){
            __syncthreads();

            if((int)threadIdx.x-stride>=0 && Col < integrcols && disp < ndisp )
                sum = slice_sm[threadIdx.x] + slice_sm[threadIdx.x-stride];
                       

            __syncthreads();
            if((int)threadIdx.x-stride>=0 && Col < integrcols && disp < ndisp )
                slice_sm[threadIdx.x] = sum;

             
        }


    if( Col<integrcols && disp < ndisp){
        integral_vol[disp*integrrows*integrcols+Row*integrcols+Col] = slice_sm[threadIdx.x];
    }
  
          
}


template<typename T>
__global__ void IntegralKernel(T* output, const int dim1 , const int dim2 , const int ndisp,const int offset){

    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* slice_sm = reinterpret_cast<T *>(shared_mem);    
    
    int Col = threadIdx.x+offset; 
    int Row = blockIdx.y; 
    int disp = blockIdx.z; 

    T val=0,temp=0,temp1=0;

    if( Col < dim2){ 
        val = output[disp*dim1*dim2+Row*dim2+Col];
    

	    for(int i=1; i<32; i<<=1 ){
	    	temp = __shfl_up(val,i);
	    	if(  (threadIdx.x & 31)  >=i )
	    		val +=temp;

	    }

	    if( (threadIdx.x &  31) ==31 || Col==(dim2-1) )
	    	slice_sm[threadIdx.x/32] = val;

    }

    __syncthreads();

    temp=0;

    if( threadIdx.x < 32 ){
    	temp = slice_sm[threadIdx.x];

    	for(int i=1; i<32; i<<=1){
    		temp1 = __shfl_up(temp,i);
    		if( (threadIdx.x & 31)   >=i )
    			temp += temp1;
    	}

    	slice_sm[threadIdx.x] = temp;

    }

	__syncthreads();

	if( Col < dim2){ 
		if(threadIdx.x >=32)
			val += slice_sm[threadIdx.x/32-1];
	
		output[disp*dim1*dim2+Row*dim2+Col] = val;
	}

  
          
}



__device__ void sort(float *x, int n)
{
	for (int i = 0; i < n - 1; i++) {
		int min = i;
		for (int j = i + 1; j < n; j++) {
			if (x[j] < x[min]) {
				min = j;
			}
		}
		float tmp = x[min];
		x[min] = x[i];
		x[i] = tmp;
	}
}

#define INDEX_D(dim0, dim1, dim2, dim3) \
	assert((dim1) >= 0 && (dim1) < size1 && (dim2) >= 0 && (dim2) < size2 && (dim3) >= 0 && (dim3) < size3), \
	((((dim0) * size3 + (dim3)) * size1 + (dim1)) * size2 + dim2)

#define INDEX(dim0, dim1, dim2, dim3) \
	assert((dim1) >= 0 && (dim1) < size1 && (dim2) >= 0 && (dim2) < size2 && (dim3) >= 0 && (dim3) < size3), \
	((((dim0) * size1 + (dim1)) * size2 + (dim2)) * size3 + dim3)


template <int sgm_direction,typename T>
__global__ void sgm_loop(float *x0, float *x1, T *input, T *output, float *tmp, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step)
{
	int x, y, dx, dy;
	int d = threadIdx.x;

	if (sgm_direction == 0) {
		/* right */
		x = step; //step;
		y = blockIdx.x;
		dx = 1;
		dy = 0;
	} else if (sgm_direction == 1) {
		/* left */
		x = size2 - 1 - step; //step;
		y = blockIdx.x;
		dx = -1;
		dy = 0;
	} else if (sgm_direction == 2) {
		/* down */
		x = blockIdx.x;
		y = step;//step;
		dx = 0;
		dy = 1;
	} else if (sgm_direction == 3) {
		/* up */
		x = blockIdx.x;
		y = size1 - 1 - step; //step;
		dx = 0;
		dy = -1;
	}

	

	if (y - dy < 0 || y - dy >= size1 || x - dx < 0 || x - dx >= size2) {
		float val = input[INDEX(0, y, x, d)];
		output[INDEX(0, y, x, d)] += val;
		tmp[d * size2 + blockIdx.x] = val;
		return;
	}

	extern __shared__ float sgm_shared[];
	float * output_s = &sgm_shared[0];
	float * output_min= &sgm_shared[size3];

	output_s[d] = output_min[d] = tmp[d * size2 + blockIdx.x];
	__syncthreads();

	for (int i = 256; i > 0; i /= 2) {
		if (d < i && d + i < size3 && output_min[d + i] < output_min[d]) {
			output_min[d] = output_min[d + i];
		}
		__syncthreads();
	}

	int ind2 = y * size2 + x;
	float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * size2 - dx);
	float D2;
	int xx = x + d * direction;
	if (xx < 0 || xx >= size2 || xx - dx < 0 || xx - dx >= size2) {
		D2 = 10;
	} else {
		D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * size2 - dx);
	}
	float P1, P2;
	if (D1 < tau_so && D2 < tau_so) {
		P1 = pi1;
		P2 = pi2;
	} else if (D1 > tau_so && D2 > tau_so) {
		P1 = pi1 / (sgm_q1 * sgm_q2);
		P2 = pi2 / (sgm_q1 * sgm_q2);
	} else {
		P1 = pi1 / sgm_q1;
		P2 = pi2 / sgm_q1;
	}

	float cost = min(output_s[d], output_min[0] + P2);
	if (d - 1 >= 0) {
		cost = min(cost, output_s[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
	}
	if (d + 1 < size3) {
		cost = min(cost, output_s[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
	}

	float val = (input[INDEX(0, y, x, d)] + cost - output_min[0]);
	output[INDEX(0, y, x, d)] += val;
	tmp[d * size2 + blockIdx.x] = val;
}


template <int sgm_direction>
__global__ void sgm2(uint8 *x0, uint8 *x1, float *input, float *output, float *tmp, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step)
{
	int x, y, dx, dy;
	int d = threadIdx.x;

	if (sgm_direction == 0) {
		/* right */
		x = blockIdx.y; //step;
		y = blockIdx.x;
		dx = 1;
		dy = 0;
	} else if (sgm_direction == 1) {
		/* left */
		x = size2 - 1 - blockIdx.y; //step;
		y = blockIdx.x;
		dx = -1;
		dy = 0;
	} else if (sgm_direction == 2) {
		/* down */
		x = blockIdx.x;
		y = blockIdx.y;//step;
		dx = 0;
		dy = 1;
	} else if (sgm_direction == 3) {
		/* up */
		x = blockIdx.x;
		y = size1 - 1 - blockIdx.y; //step;
		dx = 0;
		dy = -1;
	}

	if (y - dy < 0 || y - dy >= size1 || x - dx < 0 || x - dx >= size2) {
		float val = input[INDEX(0, y, x, d)];
		output[INDEX_D(0, y, x, d)] += val;
		tmp[d * size2 + blockIdx.x] = val;
		return;
	}

	__shared__ double output_s[400], output_min[400];

	output_s[d] = output_min[d] = tmp[d * size2 + blockIdx.x];
	__syncthreads();

	for (int i = 256; i > 0; i /= 2) {
		if (d < i && d + i < size3 && output_min[d + i] < output_min[d]) {
			output_min[d] = output_min[d + i];
		}
		__syncthreads();
	}

	int ind2 = y * size2 + x;
	float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * size2 - dx);
	float D2;
	int xx = x + d * direction;
	if (xx < 0 || xx >= size2 || xx - dx < 0 || xx - dx >= size2) {
		D2 = 10;
	} else {
		D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * size2 - dx);
	}
	float P1, P2;
	if (D1 < tau_so && D2 < tau_so) {
		P1 = pi1;
		P2 = pi2;
	} else if (D1 > tau_so && D2 > tau_so) {
		P1 = pi1 / (sgm_q1 * sgm_q2);
		P2 = pi2 / (sgm_q1 * sgm_q2);
	} else {
		P1 = pi1 / sgm_q1;
		P2 = pi2 / sgm_q1;
	}

	float cost = min(output_s[d], output_min[0] + P2);
	if (d - 1 >= 0) {
		cost = min(cost, output_s[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
	}
	if (d + 1 < size3) {
		cost = min(cost, output_s[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
	}

	float val = (input[INDEX(0, y, x, d)] + cost - output_min[0])*.25;
	output[INDEX_D(0, y, x, d)] += val;
	tmp[d * size2 + blockIdx.x] = val;
}


__global__ void cross(float *x0, float *out, int size, int dim2, int dim3, int L1, float tau1)
{

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dir = id;
		int x = dir % dim3;
		dir /= dim3;
		int y = dir % dim2;
		dir /= dim2;

		int dx = 0;
		int dy = 0;
		if (dir == 0) {
			dx = -1;
		} else if (dir == 1) {
			dx = 1;
		} else if (dir == 2) {
			dy = -1;
		} else if (dir == 3) {
			dy = 1;
		} else {
			assert(0);
		}

		int xx, yy, ind1, ind2, dist;
		ind1 = y * dim3 + x;
		for (xx = x + dx, yy = y + dy;;xx += dx, yy += dy) {
			if (xx < 0 || xx >= dim3 || yy < 0 || yy >= dim2) break;

			
			dist = max(abs(xx - x), abs(yy - y));

			if (dist == 1) continue;

			ind2 = yy * dim3 + xx;

			/* rule 1 */
			if (COLOR_DIFF(x0, ind1, ind2) >= tau1) break;

			/* rule 2 */
			if (dist >= L1) break;
		}
		out[id] = dir <= 1 ? xx : yy;

	}
}

template<typename T>
__global__ void cbca(float *x0c, float *x1c, T *vol, T *out, int size, int dim2, int dim3, int direction)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = id;
		int x = d % dim3;
		d /= dim3;
		int y = d % dim2;
		d /= dim2;

		if (x + d * direction < 0 || x + d * direction >= dim3) {
			out[id] = vol[id];
		} else {
			float sum = 0;
			int cnt = 0;

			int yy_s = max(x0c[(2 * dim2 + y) * dim3 + x], x1c[(2 * dim2 + y) * dim3 + x + d * direction]);
			int yy_t = min(x0c[(3 * dim2 + y) * dim3 + x], x1c[(3 * dim2 + y) * dim3 + x + d * direction]);
			for (int yy = yy_s + 1; yy < yy_t; yy++) {
				int xx_s = max(x0c[(0 * dim2 + yy) * dim3 + x], x1c[(0 * dim2 + yy) * dim3 + x + d * direction] - d * direction);
				int xx_t = min(x0c[(1 * dim2 + yy) * dim3 + x], x1c[(1 * dim2 + yy) * dim3 + x + d * direction] - d * direction);
				for (int xx = xx_s + 1; xx < xx_t; xx++) {
					float val = vol[(d * dim2 + yy) * dim3 + xx];
					assert(!isnan(val));
					sum += val;
					cnt++;
				}
			}

			assert(cnt > 0);
			out[id] = sum / cnt;
			assert(!isnan(out[id]));
		}
	}
}

template <typename T>
__global__ void subpixel_enchancement(float *d0, T *c2, float *out, int size, int dim23, int disp_max) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = d0[id];
		out[id] = d;
		if (1 <= d && d < disp_max - 1) {
			float cn = c2[(d - 1) * dim23 + id];
			float cz = c2[d * dim23 + id];
			float cp = c2[(d + 1) * dim23 + id];
			float denom = 2 * (cp + cn - 2 * cz);
			if (denom > 1e-5) {
				out[id] = d - min(1.0, max(-1.0, (cp - cn) / denom));
			}
		}
	}
}

__global__ void median2d(float *img, float *out, int size, int dim2, int dim3, int kernel_radius)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float xs[11 * 11];
		int xs_size = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2) {
					xs[xs_size++] = img[yy * dim3 + xx];
				}
			}
		}
		sort(xs, xs_size);
		out[id] = xs[xs_size / 2];
	}
}

__global__ void mean2d(float *img, float *kernel, float *out, int size, int kernel_radius, int dim2, int dim3, float alpha2)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float sum = 0;
		float cnt = 0;
		int i = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++, i++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2 && abs(img[yy * dim3 + xx] - img[y * dim3 + x]) < alpha2) {
					sum += img[yy * dim3 + xx] * kernel[i];
					cnt += kernel[i];
				}
			}
		}
		out[id] = sum / cnt;
	}
}



